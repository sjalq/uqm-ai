/* cffi cdef for libmelee.so - keep in sync with ai_bridge.h */

typedef struct {
    uint8_t *pixels;
    int width;
    int height;
    float reward_p1;
    float reward_p2;
    int done;
    int p1_crew;
    int p2_crew;
    int p1_max_crew;
    int p2_max_crew;
    int p1_energy;
    int p2_energy;
    int winner;
    int frame_count;
} MeleeStepResult;

int melee_lib_init(void);
void melee_lib_shutdown(void);
int melee_init(int ship_p1, int ship_p2, int p2_cyborg, int headless, uint32_t seed);
MeleeStepResult melee_step(uint8_t p1_action, uint8_t p2_action);
void melee_close(void);
int melee_get_ship_count(void);
const char* melee_get_ship_name(int index);
int melee_is_active(void);
