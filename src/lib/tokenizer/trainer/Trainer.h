#ifndef TRAINER_H
#define TRAINER_H

#include <vector>

namespace llm_fs::tokenizer::trainer {
    struct Node {
        int v;
        int block;
        Node *l;
        Node *r;
        Node *prev_occ;
        Node *next_occ;
        struct Pair *pair;
    };

    struct LinkedList {
        Node *start;
        int size;

        LinkedList(Node *s, int sz) : start(s), size(sz) {
        }
    };

    struct Pair {
        int a, b, num_occurrences;
        Node *first_occurrence, *last_occurrence;
        Pair *next, *prev;
    };

    struct HashEntry {
        int prev_connector_l = -1;
        int prev_connector_r = -1;
        Pair *ref_l = nullptr;
        Pair *ref_r = nullptr;
    };

    struct Merge {
        int first_id, second_id, token_id;
        int token_list_len;
        std::vector<int> token_list;
    };


    class Trainer {
    protected:
        static void removeFromList(LinkedList &l, const Node *node);

        static void addToHeap(std::vector<Pair *> &heap, Pair *p);

        static void removeFromHeap(std::vector<Pair *> &heap, Pair *p);

        static void removeOccFromPair(Pair *p, Node *node);

        static void removeOcc(std::vector<Pair *> &heap, Node *node);

    public:
        static std::vector<Merge> train(const std::vector<int> &ids, int num_tokens, int init_tokens);
    };
}


#endif //TRAINER_H
