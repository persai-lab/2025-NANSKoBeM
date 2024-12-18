import torch
import torch.nn as nn


class NANS_KoBeM(nn.Module):
    """
    Extension of Memory-Augmented Neural Network (MANN)
    """

    def __init__(self, config):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(config.gpu_device)
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")
        self.metric = config.metric

        # initialize the parameters
        self.num_questions = config.num_items
        self.embedding_size_q = config.embedding_size_q
        self.embedding_size_a = config.embedding_size_a
        self.num_concepts = config.num_concepts
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.summary_dim = config.summary_dim
        self.key_matrix = torch.Tensor(self.num_concepts, self.key_dim).to(self.device)
        self.init_std = config.init_std
        nn.init.normal_(self.key_matrix, mean=0, std=self.init_std)

        self.value_matrix_init = torch.Tensor(self.num_concepts, self.value_dim).to(self.device)
        nn.init.normal_(self.value_matrix_init, mean=0., std=self.init_std)

        # self.value_matrix = None

        # initialize the layers
        self.q_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + 1,
                                           embedding_dim=self.embedding_size_q,
                                           padding_idx=0)

        if self.metric == "rmse":
            self.a_embed_matrix = nn.Linear(1, self.embedding_size_a)
        else:
            self.a_embed_matrix = nn.Embedding(3, self.embedding_size_a, padding_idx=2)


        self.mapQ_key = nn.Linear(self.embedding_size_q, self.key_dim, bias=True)
        self.erase_E_Q = nn.Linear(self.embedding_size_q + self.embedding_size_a, self.value_dim, bias=True)
        self.add_D_Q = nn.Linear(self.embedding_size_q + self.embedding_size_a, self.value_dim, bias=True)

        self.erase_linear = nn.Linear(self.value_dim, self.value_dim)
        self.add_linear = nn.Linear(self.value_dim, self.value_dim)
        self.summary_fc = nn.Linear(self.embedding_size_q + self.value_dim, self.summary_dim)
        self.linear_out = nn.Linear(self.summary_dim, 1)

        self.summary_fc_behavior = nn.Linear(self.embedding_size_q + self.value_dim, self.summary_dim)
        self.linear_out_behavior = nn.Linear(self.summary_dim, self.num_questions+1)

        # initialize the activiate functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, q_data, a_data):
        """
        get output of the model with size (batch_size, seq_len)
        :param q_data: (batch_size, seq_len)
        :param qa_data: (batch_size, seq_len)
        :return:
        """

        batch_size, seq_len = q_data.size(0), q_data.size(1)
        self.value_matrix = self.value_matrix_init.clone().repeat(batch_size, 1, 1)

        # get embeddings of learning material and response
        q_embed_data = self.q_embed_matrix(q_data.long())
        if self.metric == 'rmse':
            a_data = torch.unsqueeze(a_data, dim=2)
            a_embed_data = self.a_embed_matrix(a_data)
        else:
            a_embed_data = self.a_embed_matrix(a_data)

        # split the data seq into chunk and process sequentially
        sliced_q_embed_data = torch.chunk(q_embed_data, seq_len, dim=1)
        sliced_a_embed_data = torch.chunk(a_embed_data, seq_len, dim=1)

        batch_pred, batch_pred_material = [], []

        for i in range(seq_len - 1):
            q = sliced_q_embed_data[i].squeeze(1)  # (batch_size, key_dim)
            a = sliced_a_embed_data[i].squeeze(1)  # (batch_size, key_dim)

            # hidden knowledge transfer
            qa = torch.cat([q, a], dim=1)

            q_read_key = self.mapQ_key(q)

            correlation_weight = self.compute_correlation_weight(q_read_key)
            self.value_matrix = self.write(correlation_weight, qa)

            #material prediction layer
            read_content = self.read(correlation_weight)
            behavior_level = torch.cat([read_content, q], dim=1)
            summary_output_behavior = self.tanh(self.summary_fc_behavior(behavior_level))
            batch_sliced_pred_material = self.sigmoid(self.linear_out_behavior(summary_output_behavior))
            batch_pred_material.append(batch_sliced_pred_material)

            # response prediction layer
            q_next = sliced_q_embed_data[i + 1].squeeze(1)  # (batch_size, key_dim)
            q_read_key_next = self.mapQ_key(q_next)
            correlation_weight_next = self.compute_correlation_weight(q_read_key_next)
            read_content_next = self.read(correlation_weight_next)

            mastery_level = torch.cat([read_content_next, q_next], dim=1)
            summary_output = self.tanh(self.summary_fc(mastery_level))
            batch_sliced_pred = self.sigmoid(self.linear_out(summary_output))
            batch_pred.append(batch_sliced_pred)
        batch_pred = torch.cat(batch_pred, dim=-1)
        batch_pred_material = torch.stack(batch_pred_material, dim=1)
        return batch_pred, batch_pred_material




    def compute_correlation_weight(self, query_embedded):
        """
        use dot product to find the similarity between question embedding and each concept
        embedding stored as key_matrix
        where key-matrix could be understood as all concept embedding covered by the course.

        query_embeded : (batch_size, concept_embedding_dim)
        key_matrix : (num_concepts, concept_embedding_dim)
        output: is the correlation distribution between question and all concepts
        """

        similarity = query_embedded @ self.key_matrix.t()
        correlation_weight = torch.softmax(similarity, dim=1)
        return correlation_weight

    def read(self, correlation_weight):
        """
        read function is to read a student's knowledge level on part of concepts covered by a
        target question.
        we could view value_matrix as the latent representation of a student's knowledge
        in terms of all possible concepts.

        value_matrix: (batch_size, num_concepts, concept_embedding_dim)
        correlation_weight: (batch_size, num_concepts)
        """
        batch_size = self.value_matrix.size(0)
        value_matrix_reshaped = self.value_matrix.reshape(
            batch_size * self.num_concepts, self.value_dim
        )
        correlation_weight_reshaped = correlation_weight.reshape(batch_size * self.num_concepts, 1)
        # a (10,3) * b (10,1) = c (10, 3)is every row vector of a multiplies the row scalar of b
        # the multiplication below is to scale the memory embedding by the correlation weight
        rc = value_matrix_reshaped * correlation_weight_reshaped
        read_content = rc.reshape(batch_size, self.num_concepts, self.value_dim)
        read_content = torch.sum(read_content, dim=1)  # sum over all concepts

        return read_content

    def write(self, correlation_weight, interaction_embedded):
        """
        write function is to update memory based on the interaction
        value_matrix: (batch_size, memory_size, memory_state_dim)
        correlation_weight: (batch_size, memory_size)
        qa_embedded: (batch_size, memory_state_dim)
        """
        batch_size = self.value_matrix.size(0)
        erase_vector = self.erase_linear(self.erase_E_Q(interaction_embedded))  # (batch_size, memory_state_dim)
        erase_signal = self.sigmoid(erase_vector)

        add_vector = self.add_linear(self.add_D_Q(interaction_embedded))  # (batch_size, memory_state_dim)
        add_signal = self.tanh(add_vector)

        erase_reshaped = erase_signal.reshape(batch_size, 1, self.value_dim)
        cw_reshaped = correlation_weight.reshape(batch_size, self.num_concepts, 1)
        # the multiplication is to generate weighted erase vector for each memory cell
        # therefore, the size is (batch_size, memory_size, memory_state_dim)
        erase_mul = erase_reshaped * cw_reshaped
        memory_after_erase = self.value_matrix * (1 - erase_mul)

        add_reshaped = add_signal.reshape(batch_size, 1, self.value_dim)
        # the multiplication is to generate weighted add vector for each memory cell
        # therefore, the size is (batch_size, memory_size, memory_state_dim)
        add_memory = add_reshaped * cw_reshaped

        updated_memory = memory_after_erase + add_memory
        return updated_memory
