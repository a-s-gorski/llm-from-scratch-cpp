#include "Trainer.h"

namespace llm_fs::tokenizer::trainer {
    void Trainer::removeFromList(LinkedList &l, const Node *node) {
        if (node->l != nullptr) {
            node->l->r = node->r;
        } else {
            l.start = node->r; // Corrected assignment
        }
        if (node->r != nullptr) {
            node->r->l = node->l;
        }
    }

    void Trainer::addToHeap(std::vector<Pair *> &heap, Pair *p) {
        if (const int index = p->num_occurrences; index > 0) {
            if (heap[index] == nullptr) {
                heap[index] = p;
            } else {
                p->next = heap[index];
                heap[index]->prev = p;
                heap[index] = p;
            }
        }
    }

    void Trainer::removeFromHeap(std::vector<Pair *> &heap, Pair *p) {
        if (p->prev != nullptr) {
            p->prev->next = p->next;
        } else {
            heap[p->num_occurrences] = p->next;
        }
        if (p->next != nullptr) {
            p->next->prev = p->prev;
        }
        p->next = nullptr;
        p->prev = nullptr;
    }

    void Trainer::removeOccFromPair(Pair *p, Node *node) {
        if (node->next_occ != nullptr) {
            node->next_occ->prev_occ = node->prev_occ;
        }
        if (node->prev_occ != nullptr) {
            node->prev_occ->next_occ = node->next_occ;
        }
        if (p->first_occurrence == node) {
            p->first_occurrence = node->next_occ;
        }
        if (p->last_occurrence == node) {
            p->last_occurrence = node->prev_occ;
        }
        node->prev_occ = nullptr;
        node->next_occ = nullptr;
        node->pair = nullptr;
        p->num_occurrences--;
    }

    void Trainer::removeOcc(std::vector<Pair *> &heap, Node *node) {
        if (node != nullptr && node->pair != nullptr) {
            Pair *this_pair = node->pair;
            removeFromHeap(heap, this_pair);
            removeOccFromPair(this_pair, node);
            addToHeap(heap, this_pair);
        }
    }

    std::vector<Merge> Trainer::train(const std::vector<int> &ids, const int num_tokens, const int init_tokens) {
        const unsigned int k = num_tokens - init_tokens;
        std::vector<Merge> vocab(init_tokens + k);

        for (int i = 0; i < init_tokens + k; i++) {
            vocab[i] = {i, i, i, 1, {i}};
        }

        std::vector<Node> nodeMalloc(ids.size());
        LinkedList l(&nodeMalloc[0], ids.size());
        Node *curr = nullptr;

        std::vector<Pair> init_counts(init_tokens * init_tokens);
        for (int i = 0; i < init_tokens; i++) {
            for (int j = 0; j < init_tokens; j++) {
                unsigned int index = i * init_tokens + j;
                init_counts[index] = {i, j, 0, nullptr, nullptr, nullptr, nullptr};
            }
        }

        for (size_t i = 0; i < ids.size(); i++) {
            nodeMalloc[i] = {ids[i], static_cast<int>(i), curr, nullptr, nullptr, nullptr, nullptr};
            Node *this_node = &nodeMalloc[i];
            if (ids[i] == 0) {
                this_node->block = -1;
            }
            if (curr != nullptr) {
                curr->r = this_node;
                int a = ids[i - 1];
                int b = ids[i];

                if (this_node->block != -1 && curr->block != -1) {
                    const int index = a * init_tokens + b;
                    Pair *curr_pair = &init_counts[index];
                    curr_pair->num_occurrences++;
                }
            }
            curr = this_node;
        }

        int max_occ = 0;
        for (const auto &p: init_counts) {
            if (p.num_occurrences > max_occ) {
                max_occ = p.num_occurrences;
            }
        }
        max_occ++;

        std::vector<Pair *> heap(max_occ, nullptr);
        int heap_max = max_occ - 1;

        for (auto &p: init_counts) {
            addToHeap(heap, &p);
        }

        std::vector<HashEntry> hashes(init_tokens + k + 1);
        for (int i = 0; i < k; i++) {
            while (heap_max > 0 && heap[heap_max] == nullptr) {
                heap_max--;
            }
            if (heap_max <= 0) {
                vocab[0].token_id = -1;
                break;
            }
            Pair *max_pair = heap[heap_max];
            const int new_letter = init_tokens + i;

            vocab[new_letter] = {
                max_pair->a, max_pair->b, new_letter,
                vocab[max_pair->a].token_list_len + vocab[max_pair->b].token_list_len, {}
            };
            vocab[new_letter].token_list.insert(vocab[new_letter].token_list.end(),
                                                vocab[max_pair->a].token_list.begin(),
                                                vocab[max_pair->a].token_list.end());
            vocab[new_letter].token_list.insert(vocab[new_letter].token_list.end(),
                                                vocab[max_pair->b].token_list.begin(),
                                                vocab[max_pair->b].token_list.end());

            removeFromHeap(heap, max_pair);
        }

        return vocab;
    }
}
