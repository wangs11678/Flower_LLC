#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "liblinear\linear.h"

void SVM_predict(int argc, char **argv, char *testFile, char *modelFile, char *resultFile);