#include "time_ms.h"
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
unsigned long time_ms()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (unsigned long)tv.tv_sec*1000ul + (unsigned long)tv.tv_usec/1000ul;
}
