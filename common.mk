CC=gcc
AR=ar

BASE_DIR=$(shell pwd)
LIB_DIR=$(BASE_DIR)/lib
TEST_DIR=$(BASE_DIR)/test
BIN_DIR=$(BASE_DIR)/bin

IFLAGS=-I$(BASE_DIR)/include

DEBUG=1

ifeq ($(DEBUG), 0)
CFLAGS=-O3 -Wall -Wno-unused-function -MMD -MP
else
CFLAGS=-O0 -Wall -Wno-unused-function -MMD -MP -g3 -DDEBUG
endif

LDFLAGS=-flto -L$(LIB_DIR)

