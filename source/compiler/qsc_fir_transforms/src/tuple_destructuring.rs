// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Tuple-destructuring normalization — rewrites body-local tuple-destructuring
//! `let`s into positional field projections so a whole-value source local's
//! only uses become field accesses.
//!
//! This runs at the top of each iteration of both fixed-point loops that
//! consume its output: [`crate::tuple_decompose::tuple_decompose`] and
//! [`crate::arg_promote::promote_to_fixed_point`]. By turning `let (a, b) =
//! src;` into `let a = src::0; let b = src::1;`, it exposes input parameters as
//! promotion candidates and makes non-parameter source locals field-only so
//! tuple-decompose can scalar-replace them.
//!
//! Synthesized expressions use [`crate::EMPTY_EXEC_RANGE`];
//! [`crate::exec_graph_rebuild`] rebuilds exec graphs later.

use crate::EMPTY_EXEC_RANGE;
use crate::fir_builder::alloc_local_var_expr;
use crate::fir_builder::reachable_local_callables;
use crate::reachability::collect_reachable_from_entry;
use crate::tuple_decompose::collect_all_block_ids_in_callable;
use qsc_data_structures::span::Span;
use qsc_fir::assigner::Assigner;
use qsc_fir::fir::{
    BlockId, Expr, ExprId, ExprKind, Field, FieldPath, ItemKind, LocalItemId, LocalVarId,
    Mutability, Package, PackageId, PackageLookup, PackageStore, PatId, PatKind, Res, Stmt, StmtId,
    StmtKind,
};
use qsc_fir::ty::Ty;

/// A pending rewrite of a tuple-destructuring `let` into positional
/// field projections, collected under a shared borrow before mutation.
struct DestructureRewrite {
    /// The block containing the destructuring statement.
    block_id: BlockId,
    /// The destructuring `let` statement to rewrite in place.
    stmt_id: StmtId,
    /// Mutability of the original `let`.
    mutability: Mutability,
    /// The source local read as a whole value on the right-hand side.
    source_local: LocalVarId,
    /// The full tuple type of the source local.
    tuple_ty: Ty,
    /// The element sub-patterns of the destructuring tuple pattern.
    element_pat_ids: Vec<PatId>,
}

/// Normalizes tuple-destructuring `let`s into positional field projections
/// so the destructured source local's only uses become field accesses.
///
/// For a statement `let (a, b, ...) = src;` where `src` is read as a bare
/// whole-value `Var(Local)` — an input-bound parameter or any other local —
/// this rewrites it into `let a = src::0; let b = src::1; ...`, emitting one
/// projection per non-discard element. After this rewrite the source local's
/// only uses are field projections, which:
/// - lets [`crate::arg_promote`]'s candidate search treat an input parameter
///   as a promotion candidate, and
/// - makes a non-parameter source local field-only, so
///   [`crate::tuple_decompose`] can scalar-replace it.
///
/// Only a bare `Var(Local)` right-hand side is rewritten. A `Call`, `Tuple`
/// literal, or any other RHS is left untouched, since tuple-decompose already
/// handles those once the destructured local is field-only.
///
/// Runs at the top of each iteration of both the
/// [`crate::tuple_decompose::tuple_decompose`] and
/// [`crate::arg_promote::promote_to_fixed_point`] loops, scoped to reachable
/// local callable bodies.
///
/// # Returns
///
/// `true` if any destructuring rewrite was applied; `false` otherwise.
///
/// # Element handling
///
/// Each destructuring element is recursively descended to its `Bind` leaves,
/// threading a cumulative positional index path. Every leaf emits a single
/// direct multi-index projection — no intermediate whole-value temporary is
/// created for nested elements:
/// - `PatKind::Discard`: emits no binding, since the projection is a pure
///   read of an already-evaluated local.
/// - `PatKind::Bind`: emits `let <bind> = src::Path[i, ...];`, reusing the
///   existing sub-binding's `PatId` so its `LocalVarId` is preserved.
/// - `PatKind::Tuple` (nested): recurses into each child, so `(y, z)` at
///   index `i` flattens directly to `let y = src::Path[i, 0]; let z =
///   src::Path[i, 1];`.
///
/// # Mutations
/// - Rewrites the original destructuring statement in place to the first
///   emitted projection (or removes it from its block when every element is
///   a discard).
/// - Allocates fresh `Expr`/`Pat`/`Stmt` nodes (with `EMPTY_EXEC_RANGE`)
///   for the remaining projections and splices them into the block.
pub(crate) fn normalize_tuple_destructuring(
    store: &mut PackageStore,
    package_id: PackageId,
    assigner: &mut Assigner,
) -> bool {
    let reachable = collect_reachable_from_entry(store, package_id);
    let package = store.get(package_id);
    let local_item_ids: Vec<LocalItemId> =
        reachable_local_callables(package, package_id, &reachable)
            .map(|(id, _)| id)
            .collect();

    // Note: the entry callable is intentionally *not* excluded here. This pass
    // only rewrites body-local `let (a, b) = local;` destructures into positional
    // projections; it never reshapes `decl.input`. The entry input ABI is
    // protected solely by the exclusion in `find_promotion_candidates`, which is
    // the only place `decl.input` is flattened.
    let mut rewrites: Vec<DestructureRewrite> = Vec::new();
    for item_id in local_item_ids {
        let item = package.get_item(item_id);
        let ItemKind::Callable(_) = &item.kind else {
            continue;
        };

        for block_id in collect_all_block_ids_in_callable(package, item_id) {
            let block = package.get_block(block_id);
            for &stmt_id in &block.stmts {
                let stmt = package.get_stmt(stmt_id);
                let StmtKind::Local(mutability, pat_id, rhs_id) = &stmt.kind else {
                    continue;
                };
                let pat = package.get_pat(*pat_id);
                let PatKind::Tuple(element_pat_ids) = &pat.kind else {
                    continue;
                };
                let rhs = package.get_expr(*rhs_id);
                // Only normalize a bare whole-value `Var(Local)` RHS. Any other
                // RHS (call, tuple literal, ...) is handled by tuple-decompose directly.
                let ExprKind::Var(Res::Local(source_local), _) = &rhs.kind else {
                    continue;
                };
                // Only normalize when the RHS tuple arity matches the
                // destructuring pattern arity; per-leaf element types are
                // read directly from each leaf sub-pattern's `Pat.ty`.
                match &rhs.ty {
                    Ty::Tuple(elems) if elems.len() == element_pat_ids.len() => {}
                    _ => continue,
                }
                rewrites.push(DestructureRewrite {
                    block_id,
                    stmt_id,
                    mutability: *mutability,
                    source_local: *source_local,
                    tuple_ty: rhs.ty.clone(),
                    element_pat_ids: element_pat_ids.clone(),
                });
            }
        }
    }

    if rewrites.is_empty() {
        return false;
    }

    let package = store.get_mut(package_id);
    for rewrite in rewrites {
        apply_destructure_rewrite(package, assigner, &rewrite);
    }
    true
}

/// Rewrites a single tuple-destructuring statement into positional field
/// projections (see [`normalize_tuple_destructuring`]).
fn apply_destructure_rewrite(
    package: &mut Package,
    assigner: &mut Assigner,
    rewrite: &DestructureRewrite,
) {
    // Recursively descend each element pattern to its `Bind` leaves under a
    // shared borrow, collecting `(leaf_pat_id, index_path, leaf_ty)`. This
    // avoids holding the shared borrow across the mutating projection
    // helpers below.
    let mut leaves: Vec<(PatId, Vec<usize>, Ty)> = Vec::new();
    {
        let mut indices: Vec<usize> = Vec::new();
        for (i, &elem_pat_id) in rewrite.element_pat_ids.iter().enumerate() {
            indices.push(i);
            collect_leaf_projections(package, elem_pat_id, &mut indices, &mut leaves);
            indices.pop();
        }
    }

    // Build one `(mutability, pat, rhs)` projection descriptor per leaf bind.
    let mut descriptors: Vec<(Mutability, PatId, ExprId)> = Vec::with_capacity(leaves.len());
    for (leaf_pat_id, indices, leaf_ty) in leaves {
        let proj = create_local_projection_path(
            package,
            assigner,
            rewrite.source_local,
            &rewrite.tuple_ty,
            &leaf_ty,
            &indices,
        );
        descriptors.push((rewrite.mutability, leaf_pat_id, proj));
    }

    if descriptors.is_empty() {
        // Every element is a discard: drop the now-dead destructuring use of
        // the source local so it no longer blocks promotion or tuple-decompose.
        let block = package
            .blocks
            .get_mut(rewrite.block_id)
            .expect("block should exist");
        if let Some(pos) = block.stmts.iter().position(|&s| s == rewrite.stmt_id) {
            block.stmts.remove(pos);
        }
        return;
    }

    // Reuse the original statement for the first projection.
    {
        let (mutability, pat_id, rhs_id) = descriptors[0];
        let stmt = package
            .stmts
            .get_mut(rewrite.stmt_id)
            .expect("stmt should exist");
        stmt.kind = StmtKind::Local(mutability, pat_id, rhs_id);
    }

    // Allocate fresh statements for the remaining projections.
    let mut new_stmt_ids: Vec<StmtId> = Vec::with_capacity(descriptors.len() - 1);
    for &(mutability, pat_id, rhs_id) in &descriptors[1..] {
        let stmt_id = assigner.next_stmt();
        package.stmts.insert(
            stmt_id,
            Stmt {
                id: stmt_id,
                span: Span::default(),
                kind: StmtKind::Local(mutability, pat_id, rhs_id),
                exec_graph_range: EMPTY_EXEC_RANGE,
            },
        );
        new_stmt_ids.push(stmt_id);
    }

    // Splice the new statements into the block after the original.
    let block = package
        .blocks
        .get_mut(rewrite.block_id)
        .expect("block should exist");
    if let Some(pos) = block.stmts.iter().position(|&s| s == rewrite.stmt_id) {
        for (offset, new_id) in new_stmt_ids.into_iter().enumerate() {
            block.stmts.insert(pos + 1 + offset, new_id);
        }
    }
}

/// Recursively descends a destructuring element pattern to its `Bind`
/// leaves, collecting `(leaf_pat_id, index_path, leaf_ty)` for each.
///
/// `indices` carries the cumulative positional path from the source tuple to
/// the current pattern; it is pushed/popped around each child so callers see
/// it unchanged on return. `Discard` leaves contribute nothing. Each leaf's
/// type is read directly from its `Pat.ty` (set by frontend lowering and
/// preserved through earlier passes).
fn collect_leaf_projections(
    package: &Package,
    pat_id: PatId,
    indices: &mut Vec<usize>,
    leaves: &mut Vec<(PatId, Vec<usize>, Ty)>,
) {
    let pat = package.get_pat(pat_id);
    match &pat.kind {
        PatKind::Discard => {}
        PatKind::Bind(_) => {
            leaves.push((pat_id, indices.clone(), pat.ty.clone()));
        }
        PatKind::Tuple(sub_pats) => {
            for (i, &sub_pat_id) in sub_pats.iter().enumerate() {
                indices.push(i);
                collect_leaf_projections(package, sub_pat_id, indices, leaves);
                indices.pop();
            }
        }
    }
}

/// Allocates a `src::Path[indices...]` field projection expression over a
/// fresh `Var(Res::Local(src))` base carrying the full tuple type.
///
/// The multi-index `Field::Path` projects directly to a (possibly nested)
/// leaf in a single expression; downstream tuple-decompose / arg-promote field rewrites
/// decompose arbitrary-depth paths via their `remaining`-slice recursion.
///
/// # Mutations
/// - Inserts a fresh base `Var` `Expr` and a `Field` `Expr` (with
///   `EMPTY_EXEC_RANGE`) through `assigner`.
fn create_local_projection_path(
    package: &mut Package,
    assigner: &mut Assigner,
    source_local: LocalVarId,
    tuple_ty: &Ty,
    leaf_ty: &Ty,
    indices: &[usize],
) -> ExprId {
    let base_id = alloc_local_var_expr(
        package,
        assigner,
        source_local,
        tuple_ty.clone(),
        Span::default(),
    );
    let field_expr_id = assigner.next_expr();
    package.exprs.insert(
        field_expr_id,
        Expr {
            id: field_expr_id,
            span: Span::default(),
            ty: leaf_ty.clone(),
            kind: ExprKind::Field(
                base_id,
                Field::Path(FieldPath {
                    indices: indices.to_vec(),
                }),
            ),
            exec_graph_range: EMPTY_EXEC_RANGE,
        },
    );
    field_expr_id
}
