This repo contains various implementation of recommendation system algorithms for learning

To create a new implementation:
- Use jax and equinox
- Use the Noam Notation(Shape Suffix Notation) with x as the seperator for tensor variables names. (inputs_BLD becomes inputs_BxLxD)
- B = batch, S = sequence, E = embedding dim, H = heads, D = head dim, V = vocab, Dh = head_dim//2 (RoPE), BS = batch×seq flattened, Sq/Sk = query/key seq when asymmetric
- Don't add a leading x for Noam Notation
- After implementation verify that Noam Notation is followed everywhere, if not update the implementation.
- Use jaxtyping to annotate every tensor
- Pefer Einsum notation wherever applicable.
- Keep the implementation simple and illustrative for the purpose of learning
- Add explainations in comments for learning
- Understanding dims of each operation is very important explain that wherever it's not obvious
- Add historical context wherever applicable, that helps with learning and remembering
- Use uv for package management in this repo.