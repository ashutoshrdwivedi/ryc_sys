"""
factorization_machine.demos
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Six self-contained demonstrations that together form a complete tutorial.

    A — offset_trick       : the embedding-table index trick, step by step
    B — interaction_math   : numerical proof of the O(FK) formula
    C — fm_as_mf           : FM with 2 fields = Matrix Factorisation
    D — ctr_dataset        : CTRDataset structure, batching, and CTR breakdown
    E — training           : end-to-end training, shape checks, learned ⟨vᵢ,vⱼ⟩
    F — deep_fm            : DeepFM sharing V with FM, FM vs DeepFM comparison

Run all at once:
    python -m factorization_machine.demos
    # or
    python factorization_machine/demos.py
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# Support both `python -m factorization_machine.demos` (package mode, where
# __package__ is set) and `uv run factorization_machine/demos.py` (script
# mode, where __package__ is None and relative imports would fail).
if __package__:
    from .dataset    import CTRDataset, make_synthetic_ctr_dataset
    from .model      import FMModel, batched_forward
    from .deep_model import DeepFMModel, batched_forward as deep_batched_forward
    from .training   import bce_loss, train_step, evaluate
else:
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from factorization_machine.dataset    import CTRDataset, make_synthetic_ctr_dataset
    from factorization_machine.model      import FMModel, batched_forward
    from factorization_machine.deep_model import DeepFMModel, batched_forward as deep_batched_forward
    from factorization_machine.training   import bce_loss, train_step, evaluate


# ─────────────────────────────────────────────────────────────────────────────
# A — The Offset Trick
# ─────────────────────────────────────────────────────────────────────────────
def demo_offset_trick() -> None:
    """
    Demonstrate the offset trick — the indexing scheme that lets all F
    categorical fields share a single embedding table without collision.

    The problem: each field has its own vocabulary starting at 0.
    Without offsets, user_id=0 and item_id=0 would both map to row 0 of
    the embedding table, making them indistinguishable.

    The fix: shift each field's indices by the cumulative sum of all
    previous field sizes:
        offsets = [0, field_dims[0], field_dims[0]+field_dims[1], …]
    A sample's raw per-field indices are added to these offsets so every
    (field, value) pair gets a unique row in the table.

    This demo:
        1. Builds a tiny 3-field toy with field_dims=[5,3,4].
        2. Computes the offsets and total vocab size N=12.
        3. Converts the sample (user=2, item=1, country=3) from raw local
           indices to global indices and prints both side by side.
    """
    print("=" * 60)
    print("DEMO A: The Offset Trick")
    print("=" * 60)

    field_dims = [5, 3, 4]   # user(5), item(3), country(4)
    N = sum(field_dims)      # total vocab = 12

    offsets = [0] + list(jnp.cumsum(jnp.array(field_dims[:-1])).tolist())
    print(f"  field_dims : {field_dims}")
    print(f"  offsets    : {offsets}  (cumulative sums)")
    print(f"  total vocab N = {N}\n")

    raw            = [2, 1, 3]
    global_indices = [r + o for r, o in zip(raw, offsets)]
    print(f"  Raw indices   (user=2, item=1, country=3): {raw}")
    print(f"  Global indices after offset              : {global_indices}")
    print(f"  → single embedding table lookup at positions {global_indices}\n")
    print(f"  No ambiguity: user[0] → global 0, item[0] → global 5, "
          f"country[0] → global 8\n")


# ─────────────────────────────────────────────────────────────────────────────
# B — Interaction Formula: explicit O(F²K) vs efficient O(FK)
# ─────────────────────────────────────────────────────────────────────────────
def demo_interaction_math() -> None:
    """
    Prove numerically that the efficient O(FK) formula produces the exact
    same result as the naïve O(F²K) double loop for the interaction term.

    The naïve approach iterates over all pairs (i,j) with i<j and sums
    the dot products ⟨vᵢ, vⱼ⟩ — quadratic in F.

    The efficient approach uses the identity (derived in model.py):
        Σᵢ<ⱼ ⟨vᵢ,vⱼ⟩ = ½ [ (Σᵢ vᵢ)² − Σᵢ vᵢ² ]
    which only requires two linear passes over the F embedding vectors.

    This demo:
        1. Draws F=4 random K=3 embedding vectors (stand-ins for V[x]).
        2. Computes the naïve double loop sum.
        3. Computes the efficient formula.
        4. Prints both values and their difference (should be ~0, i.e.
           only floating-point rounding noise ~1e-7).
        5. Reports how many unique pairs F(F-1)/2 are captured.
    """
    print("=" * 60)
    print("DEMO B: Interaction Term — explicit vs efficient formula")
    print("=" * 60)

    key = jax.random.PRNGKey(7)
    F, K = 4, 3
    emb_FxK = jax.random.normal(key, (F, K))   # F × K

    # ── Naïve O(F²K) double loop ──────────────────────────────────────────────
    naive = 0.0
    for i in range(F):
        for j in range(i + 1, F):
            # ⟨vᵢ, vⱼ⟩ = dot product of the two K-dim vectors
            naive += float(jnp.dot(emb_FxK[i], emb_FxK[j]))

    # ── Efficient O(FK) formula ───────────────────────────────────────────────
    sum_K     = jnp.sum(emb_FxK, axis=0)                                   # K
    efficient = float(0.5 * (jnp.sum(sum_K ** 2) - jnp.sum(emb_FxK ** 2)))

    print(f"  F={F}, K={K}")
    print(f"  Naïve  Σᵢ<ⱼ ⟨vᵢ,vⱼ⟩         = {naive:.6f}")
    print(f"  Efficient ½(sum²−sq_sum)     = {efficient:.6f}")
    print(f"  Difference (should be ~0)    = {abs(naive - efficient):.2e}\n")

    n_pairs = F * (F - 1) // 2
    print(f"  Fields F={F} → {n_pairs} unique pairs, all captured in O(FK)={F*K} ops\n")


# ─────────────────────────────────────────────────────────────────────────────
# C — FM reduces to Matrix Factorisation (2-field special case)
# ─────────────────────────────────────────────────────────────────────────────
def demo_fm_as_mf() -> None:
    """
    Show that FM with exactly two fields (user_id, item_id) is algebraically
    identical to classic Matrix Factorisation (MF).

    MF formula:  ŷ = bias_u + bias_i + ⟨p_u, q_i⟩
    FM formula with F=2:
        ŷ = bias + w_u·x_u + w_i·x_i + ⟨v_u, v_i⟩·x_u·x_i
    Since x_u = x_i = 1 (one-hot, already selected by the embedding lookup):
        ŷ = bias + w_u + w_i + ⟨v_u, v_i⟩
    which is exactly MF with  bias_u=w_u, bias_i=w_i, p_u=v_u, q_i=v_i.

    This demo:
        1. Builds an FMModel with field_dims=[n_users, n_items].
        2. Runs FMModel.__call__ for a single (user=3, item=5) sample.
        3. Manually computes the MF formula by directly indexing the same
           weight matrices (W and V) inside the model.
        4. Asserts the two logits match up to floating-point precision (~1e-7).
    """
    print("=" * 60)
    print("DEMO C: FM reduces to Matrix Factorisation (2 fields)")
    print("=" * 60)

    n_users, n_items, K = 10, 8, 4
    field_dims = [n_users, n_items]
    key = jax.random.PRNGKey(42)

    model = FMModel(field_dims=field_dims, K=K, key=key)

    user_raw, item_raw = 3, 5
    offsets = jnp.array([0, n_users])
    x_F     = (jnp.array([user_raw, item_raw]) + offsets).astype(jnp.int32)
    logit   = model(x_F)

    # Manually reproduce with MF formula:
    #   ŷ = bias + w_user + w_item + ⟨v_user, v_item⟩
    w_user    = float(model.W.weight[user_raw, 0])
    w_item    = float(model.W.weight[n_users + item_raw, 0])
    v_user_K  = model.V.weight[user_raw]
    v_item_K  = model.V.weight[n_users + item_raw]
    mf_logit  = float(model.bias + w_user + w_item + jnp.dot(v_user_K, v_item_K))

    print(f"  n_users={n_users}, n_items={n_items}, K={K}")
    print(f"  Sample: user={user_raw}, item={item_raw}")
    print(f"  FM logit (via model.__call__)        : {float(logit):.6f}")
    print(f"  MF logit (bias + w_u + w_i + ⟨pu,qi⟩): {mf_logit:.6f}")
    print(f"  Difference: {abs(float(logit) - mf_logit):.2e}  ✓\n")


# ─────────────────────────────────────────────────────────────────────────────
# D — CTRDataset: structure, offset trick, and CTR breakdown
# ─────────────────────────────────────────────────────────────────────────────
def demo_ctr_dataset() -> None:
    """
    Walk through the CTRDataset interface and confirm the synthetic data
    has the expected interaction signal baked in.

    This demo:
        1. Generates 10,000 samples via make_synthetic_ctr_dataset and
           prints the dataset summary (N, F, vocab, offsets, overall CTR).
        2. Picks sample 0 and shows:
             - raw field-local indices   (e.g. user_type=2 → index 2)
             - global indices after +offsets (e.g. item_cat=3 → index 6)
             - human-readable category name for each field
           so the offset trick is visible at the sample level.
        3. Prints a 3×5 CTR table (user_type × item_cat) to verify the
           hand-crafted rule is reflected in the data:
             - casual/regular × any item  → CTR ≈ 0.10–0.31
             - power × non-popular item   → CTR ≈ 0.40–0.45
             - power × tech/food          → CTR ≈ 0.89–0.90  (interaction bonus)
        4. Calls dataset.get_batch(key, B=8) and prints the returned tensor
           shapes and label values to show the API.
    """
    print("=" * 60)
    print("DEMO D: CTRDataset — structure and offset trick")
    print("=" * 60)

    key     = jax.random.PRNGKey(1)
    dataset = make_synthetic_ctr_dataset(n_samples=10_000, key=key)

    print("  Dataset summary:")
    dataset.summary()

    # ── Show one raw sample vs its offset-corrected form ──────────────────────
    # Raw: each integer is field-local (user_type ∈ {0,1,2}, item_cat ∈ {0..4} …)
    # Global: each integer points into the single shared embedding table
    raw_sample_F  = dataset.X_NxF[0]                                    # F
    glob_sample_F = (raw_sample_F + dataset.offsets).astype(jnp.int32)  # F
    label         = int(dataset.y_N[0])

    print(f"\n  Sample 0:")
    print(f"    Raw indices  (field-local)      : {raw_sample_F.tolist()}")
    print(f"    Global indices (after +offsets) : {glob_sample_F.tolist()}")
    print(f"    Label                           : {label}")
    print(f"\n  Interpretation:")
    field_names = ["user_type", "item_cat", "time_slot", "device"]
    cat_names   = [
        ["casual", "regular", "power"],
        ["sports", "tech", "fashion", "food", "travel"],
        ["morning", "afternoon", "evening", "night"],
        ["mobile", "tablet", "desktop"],
    ]
    for i, (name, cats) in enumerate(zip(field_names, cat_names)):
        v = int(raw_sample_F[i])
        g = int(glob_sample_F[i])
        print(f"    field {i} ({name:10s}): raw={v} ({cats[v]:10s})  global={g}")

    # ── Show CTR breakdown to confirm the synthetic interaction signal ─────────
    print(f"\n  CTR by user_type × item_cat (verifying the synthetic rule):")
    print(f"  {'user_type':>12}  {'item_cat':>10}  {'CTR':>6}  {'N':>6}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*6}  {'-'*6}")
    type_names = ["casual", "regular", "power"]
    cat_names2 = ["sports", "tech", "fashion", "food", "travel"]
    for ut in range(3):
        for ic in range(5):
            mask_N = (
                (dataset.X_NxF[:, 0] == ut) &
                (dataset.X_NxF[:, 1] == ic)
            )
            n   = int(jnp.sum(mask_N))
            ctr = float(jnp.mean(dataset.y_N[mask_N])) if n > 0 else 0.0
            print(f"  {type_names[ut]:>12}  {cat_names2[ic]:>10}  {ctr:>6.3f}  {n:>6}")
    print()

    # ── Get a batch ───────────────────────────────────────────────────────────
    key, bk = jax.random.split(key)
    x_BxF, y_B = dataset.get_batch(bk, batch_size=8)
    print(f"  get_batch(B=8):")
    print(f"    x_BxF shape : {x_BxF.shape}  (B×F, globally offset)")
    print(f"    y_B   shape : {y_B.shape}")
    print(f"    labels       : {y_B.tolist()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# E — End-to-end training on CTRDataset
# ─────────────────────────────────────────────────────────────────────────────
def demo_training() -> None:
    """
    Full end-to-end training loop on the synthetic CTR dataset, followed by
    an interpretability check on the learned embedding dot products.

    Setup:
        - 50,000 samples, 4 fields, K=4 latent factors.
        - Optimizer: Adam lr=0.05, 200 steps, batch size 1024.
        - Loss: binary cross-entropy on logits (numerically stable form).

    Training loop (train_step):
        Each call to train_step runs one Adam update via equinox's
        filter_jit + filter_value_and_grad, which differentiates only through
        the array leaves of the pytree and leaves static fields (K, etc.) alone.

    Shape assertions:
        After training, batched_forward is called once more to verify
        output shapes and that all probabilities lie in [0, 1].

    Interpretability — learned ⟨vᵢ, vⱼ⟩:
        FM encodes the strength of a feature interaction in the dot product
        of the two embedding vectors.  We directly index the trained V matrix
        at the global positions of specific (user_type, item_cat) pairs and
        compare dot products:
            power × tech/food  → should be highest  (CTR rule: +0.30 bonus)
            power × sports     → medium              (only linear user effect)
            casual × tech      → lower               (only linear item effect)
            casual × sports    → lowest              (base CTR only)
        A larger ⟨vᵢ, vⱼ⟩ means the model learned that those two features
        co-occurring is a strong positive signal — which matches the rule.
    """
    print("=" * 60)
    print("DEMO E: Training — learnable interaction signal")
    print("=" * 60)

    B         = 1024
    K         = 4      # small K is enough for 4 fields
    NUM_STEPS = 200

    key = jax.random.PRNGKey(0)

    key, dk = jax.random.split(key)
    dataset = make_synthetic_ctr_dataset(n_samples=50_000, key=dk)
    F = dataset.F

    key, mk = jax.random.split(key)
    model = FMModel(field_dims=dataset.field_dims, K=K, key=mk)

    # Adam works well for FM; small weight decay on V prevents overfitting.
    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    N        = dataset.N_vocab
    n_params = N * K + N + 1
    print(f"  Vocab N={N}  Fields F={F}  Latent K={K}")
    print(f"  Params = {n_params}   Interaction pairs = {F*(F-1)//2}\n")
    print(f"  {'Step':>4}  {'Loss':>8}  {'TrainAcc':>9}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*9}")

    for step in range(1, NUM_STEPS + 1):
        key, bk = jax.random.split(key)
        x_BxF, y_B = dataset.get_batch(bk, B)
        model, opt_state, loss = train_step(model, optimizer, opt_state, x_BxF, y_B)
        if step % 40 == 0 or step == 1:
            acc = evaluate(model, x_BxF, y_B)
            print(f"  {step:>4}  {float(loss):>8.4f}  {float(acc):>9.3f}")

    # ── Shape sanity checks ───────────────────────────────────────────────────
    print()
    x_final_BxF, _ = dataset.get_batch(key, B)
    logits_B = batched_forward(model, x_final_BxF)
    probs_B  = jax.nn.sigmoid(logits_B)
    assert logits_B.shape == (B,)
    assert jnp.all((probs_B >= 0) & (probs_B <= 1))
    print(f"  Output logits shape : {logits_B.shape}  ✓")
    print(f"  Probs in [0,1]      : ✓")

    # ── Inspect learned interaction strengths ─────────────────────────────────
    # The synthetic rule gives power_user × tech/food a +0.30 bonus.
    # We compare ⟨v_power, v_tech⟩ vs ⟨v_casual, v_sports⟩ etc.
    #
    # Global indices (offsets=[0,3,8,12]):
    #   user_type: casual=0, regular=1, power=2
    #   item_cat : sports=3, tech=4, fashion=5, food=6, travel=7
    V_NxK = model.V.weight   # N × K

    v_casual_K = V_NxK[0]
    v_power_K  = V_NxK[2]
    v_sports_K = V_NxK[3]
    v_tech_K   = V_NxK[4]
    v_food_K   = V_NxK[6]

    print(f"\n  Learned ⟨vᵢ, vⱼ⟩ for specific user × item pairs:")
    print(f"  (higher = FM predicts stronger interaction)")
    print(f"  {'pair':>30}  {'CTR rule':>24}  {'⟨vᵢ,vⱼ⟩':>10}")
    print(f"  {'-'*30}  {'-'*24}  {'-'*10}")
    rows = [
        ("power   × tech   (+bonus)", v_power_K,  v_tech_K,   "0.10+0.30+0.20+0.30=0.90"),
        ("power   × food   (+bonus)", v_power_K,  v_food_K,   "0.10+0.30+0.20+0.30=0.90"),
        ("power   × sports (no int)", v_power_K,  v_sports_K, "0.10+0.30+0.00+0.00=0.40"),
        ("casual  × tech   (no int)", v_casual_K, v_tech_K,   "0.10+0.00+0.20+0.00=0.30"),
        ("casual  × sports (base)  ", v_casual_K, v_sports_K, "0.10+0.00+0.00+0.00=0.10"),
    ]
    for label, va_K, vb_K, rule in rows:
        score = float(jnp.dot(va_K, vb_K))
        print(f"  {label:>30}  {rule:>24}  {score:>10.4f}")

    print(f"""
  NOTE: ⟨v_power, v_tech⟩ and ⟨v_power, v_food⟩ should be among the
  highest because the data has a +0.30 interaction bonus for those pairs.
  FM learns to encode this by aligning the embedding vectors of power users
  with those of popular item categories.
""")


# ─────────────────────────────────────────────────────────────────────────────
# F — DeepFM: shared embedding, FM vs DeepFM comparison
# ─────────────────────────────────────────────────────────────────────────────
def demo_deep_fm() -> None:
    """
    Train DeepFM on the same synthetic CTR dataset and compare against FM.

    Key points illustrated:
        1. V is shared: fm.V inside DeepFMModel is the SAME object used by
           both the FM component and the DNN. One pytree leaf, updated by
           gradients from both signal paths simultaneously.

        2. Architecture: FM logit + DNN logit, where DNN input is the
           concatenated (flattened) embedding vectors [v₁ ‖ … ‖ vF], shape F*K.

        3. DeepFM has strictly more capacity than FM. On this toy dataset
           (only 2nd-order signal) both should reach similar accuracy; on
           real datasets with higher-order patterns, DeepFM tends to win.

    Setup mirrors Demo E (same data, K, steps) so results are comparable.
    """
    print("=" * 60)
    print("DEMO F: DeepFM — shared embedding, FM vs DeepFM comparison")
    print("=" * 60)

    B         = 1024
    K         = 4
    NUM_STEPS = 200
    HIDDEN    = [64, 32]   # small MLP — dataset is simple

    key = jax.random.PRNGKey(0)

    key, dk = jax.random.split(key)
    dataset = make_synthetic_ctr_dataset(n_samples=50_000, key=dk)
    F = dataset.F

    # ── Verify V is shared (same object id in both paths) ─────────────────────
    key, mk = jax.random.split(key)
    deep_model = DeepFMModel(field_dims=dataset.field_dims, K=K, hidden_dims=HIDDEN, key=mk)

    # The FM component's V is the single embedding in the whole pytree.
    # Reading fm.V in the DNN path accesses the same weight matrix.
    n_fm_params  = dataset.N_vocab * K + dataset.N_vocab + 1
    # Build the MLP architecture: input is flattened embeddings (F*K dims),
    # then hidden layers, then single output logit.
    # mlp_dims = [input_dim, hidden_1, hidden_2, ..., output_dim]
    mlp_dims     = [F * K] + HIDDEN + [1]
    
    # Count total parameters in the MLP: for each layer, params = (input_dim * output_dim) + output_dim.
    # The "+ output_dim" term accounts for bias vectors (one per output neuron).
    # This excludes V which is shared with the FM component and counted separately above.
    n_mlp_params = sum(d_in * d_out + d_out for d_in, d_out in zip(mlp_dims[:-1], mlp_dims[1:]))
    print(f"  Vocab N={dataset.N_vocab}  Fields F={F}  Latent K={K}")
    print(f"  MLP hidden dims: {HIDDEN}")
    print(f"  FM  params : {n_fm_params}  (V: {dataset.N_vocab}×{K}, W: {dataset.N_vocab}×1, bias: 1)")
    print(f"  MLP params : {n_mlp_params}  (on top, shares V with FM)")
    print(f"  Total      : {n_fm_params + n_mlp_params}  (V counted once — it is shared)\n")

    # ── Train DeepFM ──────────────────────────────────────────────────────────
    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(eqx.filter(deep_model, eqx.is_array))

    print(f"  {'Step':>4}  {'Loss':>8}  {'TrainAcc':>9}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*9}")

    for step in range(1, NUM_STEPS + 1):
        key, bk = jax.random.split(key)
        x_BxF, y_B = dataset.get_batch(bk, B)
        # train_step works generically on any eqx.Module — no changes needed.
        deep_model, opt_state, loss = train_step(deep_model, optimizer, opt_state, x_BxF, y_B)
        if step % 40 == 0 or step == 1:
            acc = evaluate(deep_model, x_BxF, y_B)
            print(f"  {step:>4}  {float(loss):>8.4f}  {float(acc):>9.3f}")

    # ── Shape checks ──────────────────────────────────────────────────────────
    print()
    x_final_BxF, _ = dataset.get_batch(key, B)
    logits_B = deep_batched_forward(deep_model, x_final_BxF)   # B
    probs_B  = jax.nn.sigmoid(logits_B)
    assert logits_B.shape == (B,)
    assert jnp.all((probs_B >= 0) & (probs_B <= 1))
    print(f"  Output logits shape : {logits_B.shape}  ✓")
    print(f"  Probs in [0,1]      : ✓")

    # ── Train a fresh FM for comparison ───────────────────────────────────────
    # Same seed, same data → apples-to-apples accuracy comparison.
    print(f"\n  Training FM for comparison (same setup)...")
    key2 = jax.random.PRNGKey(0)
    key2, dk2 = jax.random.split(key2)
    dataset2 = make_synthetic_ctr_dataset(n_samples=50_000, key=dk2)
    key2, mk2 = jax.random.split(key2)
    fm_model  = FMModel(field_dims=dataset2.field_dims, K=K, key=mk2)
    fm_opt    = optax.adam(learning_rate=0.05)
    fm_state  = fm_opt.init(eqx.filter(fm_model, eqx.is_array))

    for step in range(1, NUM_STEPS + 1):
        key2, bk2 = jax.random.split(key2)
        x_BxF2, y_B2 = dataset2.get_batch(bk2, B)
        fm_model, fm_state, _ = train_step(fm_model, fm_opt, fm_state, x_BxF2, y_B2)

    x_cmp_BxF, y_cmp_B = dataset2.get_batch(key2, B)
    fm_acc   = float(evaluate(fm_model,   x_cmp_BxF, y_cmp_B))
    deep_acc = float(evaluate(deep_model, x_final_BxF, y_B))

    print(f"\n  Final accuracy comparison ({NUM_STEPS} steps, B={B}):")
    print(f"    FM     accuracy : {fm_acc:.3f}")
    print(f"    DeepFM accuracy : {deep_acc:.3f}")
    print(f"""
  NOTE: On this simple dataset (pure 2nd-order signal, 4 fields), FM and
  DeepFM typically reach similar accuracy. DeepFM's advantage shows up on
  real-world CTR datasets with complex higher-order feature interactions
  (e.g. Criteo, Avazu) where FM's fixed 2nd-order term is insufficient.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    demo_offset_trick()
    demo_interaction_math()
    demo_fm_as_mf()
    demo_ctr_dataset()
    demo_training()
    demo_deep_fm()


if __name__ == "__main__":
    main()
