#include <metal_stdlib>


thread int index_2d_flat(
  const thread int &index_0,
  const thread int &index_1,
  constant int &shape_1
) {
  return index_1 + index_0 * shape_1;
}


thread int index_3d_flat(
  const thread int &index_0,
  const thread int &index_1,
  thread int &index_2,
  constant int &shape_1,
  constant int &shape_2
) {
  return index_2 + index_1 * shape_2 + index_0 * shape_2 * shape_1;
}

// --------------------------------------------------------------------------------------


kernel void nearest_neighbours_indexes_1d(
  device float *token_embeddings [[buffer(0)]],
  device float *embedding_matrix [[buffer(1)]],
  device int *outputs [[buffer(2)]],
  constant int &num_tokens [[buffer(3)]],
  constant int &vocab_size [[buffer(4)]],
  constant int &embed_dim [[buffer(5)]],
  uint2 tid [[thread_position_in_grid]]
) {

  float min_dist = 100;
  int argmin = 100;


  for (int word_index = 0; word_index != vocab_size; word_index++) {

    float dist = 0;

    for (int i = 0; i != embed_dim; i++) {
      const int index_in_embedding_matrix = index_2d_flat(word_index, i, embed_dim);
      const float val1 = embedding_matrix[index_in_embedding_matrix];
      const float val2 = token_embeddings[i];
      dist += metal::powr(val1 - val2, 2);
    }
    dist = metal::sqrt(dist);

    if (dist < min_dist) {
      min_dist = dist;
      argmin = word_index;
    }
  }

  outputs[0] = argmin;
}

kernel void nearest_neighbours_indexes_2d(
  device float *token_embeddings [[buffer(0)]],
  device float *embedding_matrix [[buffer(1)]],
  device int *outputs [[buffer(2)]],
  constant int &num_tokens [[buffer(3)]],
  constant int &vocab_size [[buffer(4)]],
  constant int &embed_dim [[buffer(5)]],
  uint2 tid [[thread_position_in_grid]]
) {


  const int index_in_sequence = tid[0];

  float min_dist = 100;
  int argmin = 100;


  for (int word_index = 0; word_index != vocab_size; word_index++) {

    float dist = 0;

    for (int i = 0; i != embed_dim; i++) {
      const int index_in_embedding_matrix = index_2d_flat(word_index, i, embed_dim);
      const int index_in_token_embeddings = index_2d_flat(index_in_sequence, i, embed_dim);
      const float val1 = embedding_matrix[index_in_embedding_matrix];
      const float val2 = token_embeddings[index_in_token_embeddings];
      dist += metal::powr(val1 - val2, 2);
    }
    dist = metal::sqrt(dist);

    if (dist < min_dist) {
      min_dist = dist;
      argmin = word_index;
    }
  }
  outputs[index_in_sequence] = argmin;

}


kernel void nearest_neighbours_indexes_3d(
  device float *token_embeddings [[buffer(0)]],
  device float *embedding_matrix [[buffer(1)]],
  device int *outputs [[buffer(2)]],
  constant int &num_tokens [[buffer(3)]],
  constant int &vocab_size [[buffer(4)]],
  constant int &embed_dim [[buffer(5)]],
  uint2 tid [[thread_position_in_grid]]
) {


  const int index_in_batch = tid[0];
  const int index_in_sequence = tid[1];

  float min_dist = 100;
  int argmin = 100;


  for (int word_index = 0; word_index != vocab_size; word_index++) {

    float dist = 0;

    for (int i = 0; i != embed_dim; i++) {
      const int index_in_embedding_matrix = index_2d_flat(word_index, i, embed_dim);
      const int index_in_token_embeddings = index_3d_flat(index_in_batch, index_in_sequence, i, num_tokens, embed_dim);
      const float val1 = embedding_matrix[index_in_embedding_matrix];
      const float val2 = token_embeddings[index_in_token_embeddings];
      dist += metal::powr(val1 - val2, 2);
    }
    dist = metal::sqrt(dist);

    if (dist < min_dist) {
      min_dist = dist;
      argmin = word_index;
    }
  }
  const int index_in_output = index_2d_flat(index_in_batch, index_in_sequence, num_tokens);
  outputs[index_in_output] = argmin;

}



// -------------------------------------------------------------------------------------


kernel void nearest_neighbours_1d(
  device float *token_embeddings [[buffer(0)]],
  device float *embedding_matrix [[buffer(1)]],
  device float *outputs [[buffer(2)]],
  constant int &num_tokens [[buffer(3)]],
  constant int &vocab_size [[buffer(4)]],
  constant int &embed_dim [[buffer(5)]],
  uint2 tid [[thread_position_in_grid]]
) {

  float min_dist = 100;
  int argmin = 100;


  for (int word_index = 0; word_index != vocab_size; word_index++) {

    float dist = 0;

    for (int i = 0; i != embed_dim; i++) {
      const int index_in_embedding_matrix = index_2d_flat(word_index, i, embed_dim);
      const float val1 = embedding_matrix[index_in_embedding_matrix];
      const float val2 = token_embeddings[i];
      dist += metal::powr(val1 - val2, 2);
    }
    dist = metal::sqrt(dist);

    if (dist < min_dist) {
      min_dist = dist;
      argmin = word_index;
    }
  }

  for (int i = 0; i != embed_dim; i++) {
    const int index_in_embedding_matrix = index_2d_flat(argmin, i, embed_dim);
    outputs[i] = embedding_matrix[index_in_embedding_matrix];
  }
}


kernel void nearest_neighbours_2d(
  device float *token_embeddings [[buffer(0)]],
  device float *embedding_matrix [[buffer(1)]],
  device float *outputs [[buffer(2)]],
  constant int &num_tokens [[buffer(3)]],
  constant int &vocab_size [[buffer(4)]],
  constant int &embed_dim [[buffer(5)]],
  uint2 tid [[thread_position_in_grid]]
) {

  const int index_in_sequence = tid[0];

  float min_dist = 100;
  int argmin = 100;


  for (int word_index = 0; word_index != vocab_size; word_index++) {

    float dist = 0;

    for (int i = 0; i != embed_dim; i++) {
      const int index_in_embedding_matrix = index_2d_flat(word_index, i, embed_dim);
      const int index_in_token_embeddings = index_2d_flat(index_in_sequence, i, embed_dim);
      const float val1 = embedding_matrix[index_in_embedding_matrix];
      const float val2 = token_embeddings[index_in_token_embeddings];
      dist += metal::powr(val1 - val2, 2);
    }
    dist = metal::sqrt(dist);

    if (dist < min_dist) {
      min_dist = dist;
      argmin = word_index;
    }
  }

  for (int i = 0; i != embed_dim; i++) {
    const int index_in_output = index_2d_flat(index_in_sequence, i, embed_dim);
    const int index_in_embedding_matrix = index_2d_flat(argmin, i, embed_dim);
    outputs[index_in_output] = embedding_matrix[index_in_embedding_matrix];
  }
}


kernel void nearest_neighbours_3d(
  device float *token_embeddings [[buffer(0)]],
  device float *embedding_matrix [[buffer(1)]],
  device float *outputs [[buffer(2)]],
  constant int &num_tokens [[buffer(3)]],
  constant int &vocab_size [[buffer(4)]],
  constant int &embed_dim [[buffer(5)]],
  uint2 tid [[thread_position_in_grid]]
) {


  const int index_in_batch = tid[0];
  const int index_in_sequence = tid[1];

  float min_dist = 100;
  int argmin = 100;


  for (int word_index = 0; word_index != vocab_size; word_index++) {

    float dist = 0;

    for (int i = 0; i != embed_dim; i++) {
      const int index_in_embedding_matrix = index_2d_flat(word_index, i, embed_dim);
      const int index_in_token_embeddings = index_3d_flat(index_in_batch, index_in_sequence, i, num_tokens, embed_dim);
      const float val1 = embedding_matrix[index_in_embedding_matrix];
      const float val2 = token_embeddings[index_in_token_embeddings];
      dist += metal::powr(val1 - val2, 2);
    }
    dist = metal::sqrt(dist);

    if (dist < min_dist) {
      min_dist = dist;
      argmin = word_index;
    }
  }

  for (int i = 0; i != embed_dim; i++) {
    const int index_in_output = index_3d_flat(index_in_batch, index_in_sequence, i, num_tokens, embed_dim);
    const int index_in_embedding_matrix = index_2d_flat(argmin, i, embed_dim);
    outputs[index_in_output] = embedding_matrix[index_in_embedding_matrix];
  }
}

