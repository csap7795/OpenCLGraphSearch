#!/bin/bash
# Get the location of this makefile.
# install realpath otherwise it won't work
ROOT_DIR := $(shell dirname "$(realpath $(lastword $(MAKEFILE_LIST)))")
CFLAGS = -Wall -g -fPIC -std=c99
LD_FLAGS = -lGraphSearch -lOpenCL -lm
OBJDIR = obj/
MY_INCLUDES = /headers/# define user's include directory path
C_FILES := $(wildcard sources/*.c)
OBJ_FILES := $(addprefix obj/,$(notdir $(C_FILES:.c=.o)))
INCLUDES := -I/usr/local/include -I/usr/include/CL

myprog: libGraphSearch.so
	LD_LIBRARY_PATH="$(ROOT_DIR)/bin/Debug/" gcc $(CFLAGS) -I"$(ROOT_DIR)$(MY_INCLUDES)" "$(ROOT_DIR)/main.c" -o myprog -L"$(ROOT_DIR)/bin/Debug" $(LD_FLAGS)

libGraphSearch.so: $(OBJ_FILES)
	gcc -shared -o bin/Debug/libGraphSearch.so $(OBJDIR)*.o -lOpenCL -lm

obj/%.o: sources/%.c
	gcc $(CFLAGS) -I"$(ROOT_DIR)$(MY_INCLUDES)" $(INCLUDES) -c -o "$(ROOT_DIR)/$@" "$(ROOT_DIR)/$<"

