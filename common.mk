CC=gcc
CPPC=g++
AR=ar

BASE_DIR=$(shell pwd)
LIB_DIR=$(BASE_DIR)/lib
TEST_DIR=$(BASE_DIR)/test
BIN_DIR=$(BASE_DIR)/bin

IFLAGS=-I$(BASE_DIR)/include

DEBUG=0

ifeq ($(DEBUG), 0)
CFLAGS=-O3 -g -Wall -Wno-unused-function -MMD -MP -mavx -DCL_SILENCE_DEPRECATION
CPPFLAGS=-O3 -g -Wall -Wno-unused-function -MMD -MP -mavx -DCL_SILENCE_DEPRECATION
else
CFLAGS=-O0 -g -Wall -Wno-unused-function -MMD -MP -mavx -g3 -DDEBUG -DCL_SILENCE_DEPRECATION
CPPFLAGS=-O0 -g -Wall -Wno-unused-function -MMD -MP -mavx -g3 -DDEBUG -DCL_SILENCE_DEPRECATION
endif

LDFLAGS=-flto -L$(LIB_DIR) -framework OpenCL

