#include <stdint.h>

#include "model.h"
#include "test_data.h"

#ifndef MODEL_MAIN_USE_STDIO
#define MODEL_MAIN_USE_STDIO 1
#endif

#if MODEL_MAIN_USE_STDIO
#include <stdio.h>
#endif

#if MODEL_IN_CH != TEST_DATA_IN_CH
#error "MODEL_IN_CH and TEST_DATA_IN_CH must match"
#endif

#if MODEL_CLASSES != TEST_DATA_NUM_CLASSES
#error "MODEL_CLASSES and TEST_DATA_NUM_CLASSES must match"
#endif

#ifndef TEST_DATA_HAS_LOGITS_Q
#define TEST_DATA_HAS_LOGITS_Q 0
#endif

static tcn_t net;

static void logits_to_pred(const int16_t *logits, uint8_t *pred)
{
    for (int i = 0; i < MODEL_CLASSES; ++i) {
        pred[i] = (logits[i] >= 0) ? 1u : 0u;
    }
}

static int run_record(int rec, int16_t *logits_out)
{
    (void)rec;
    if (tcn_init(&net, &model) != 0) return -1;

    for (int t = 0; t < TEST_DATA_SEQ_LEN; ++t) {
        tcn_step(&net, test_data_xq[rec][t], logits_out);
    }

    return 0;
}

static double class_f1(int tp, int fp, int fn)
{
    const int den_p = tp + fp;
    const int den_r = tp + fn;
    double p = 0.0;
    double r = 0.0;

    if (den_p > 0) p = (double)tp / (double)den_p;
    if (den_r > 0) r = (double)tp / (double)den_r;
    if ((p + r) <= 0.0) return 0.0;
    return 2.0 * p * r / (p + r);
}

static double macro_f1_score(const int *tp, const int *fp, const int *fn)
{
    double macro_f1 = 0.0;

    for (int i = 0; i < MODEL_CLASSES; ++i) {
        macro_f1 += class_f1(tp[i], fp[i], fn[i]);
    }

    return macro_f1 / (double)MODEL_CLASSES;
}

int main(void)
{
    int16_t logits[MODEL_CLASSES];
    uint8_t pred[MODEL_CLASSES];
    int short_context = 0;
    int ran = 0;
    int tp[MODEL_CLASSES] = {0};
    int fp[MODEL_CLASSES] = {0};
    int fn[MODEL_CLASSES] = {0};

    for (int rec = 0; rec < TEST_DATA_NUM_RECORDS; ++rec) {
        if (run_record(rec, logits) != 0) {
#if MODEL_MAIN_USE_STDIO
            printf("record %u failed: stream shorter than warmup\n",
                   (unsigned)test_data_record_ids[rec]);
#endif
            return 1;
        }

        if (!tcn_ready(&net)) short_context++;
        logits_to_pred(logits, pred);
        for (int i = 0; i < MODEL_CLASSES; ++i) {
            const int y = test_data_labels[rec][i] ? 1 : 0;
            const int p = pred[i] ? 1 : 0;
            if (p && y) tp[i]++;
            else if (p && !y) fp[i]++;
            else if (!p && y) fn[i]++;
        }
        ran++;
    }

#if MODEL_MAIN_USE_STDIO
    if (ran == 0) {
        printf("No test records\n");
        return 1;
    }

    {
        const double macro_f1 = macro_f1_score(tp, fp, fn);
        printf("summary records=%d", ran);
        if (short_context > 0) {
            printf(" short_context=%d", short_context);
        }
        printf("\n");
        printf("summary metric=macro_f1 threshold=logit>=0 value=%.4f\n", macro_f1);
    }
#else
    (void)ran;
#endif

    return 0;
}
