#pragma once

#ifdef BATCH_SIZE
const int batch_size = BATCH_SIZE;
#else
const int batch_size = 1;
#endif

#ifdef IN_CHANNELS
const int in_channels = IN_CHANNELS;
#else
const int in_channels = 32;
#endif

#ifdef IN_HEIGHT
const int in_height = IN_HEIGHT;
#else
const int in_height = 32;
#endif

#ifdef IN_WIDTH
const int in_width = IN_WIDTH;
#else
const int in_width = 32;
#endif
