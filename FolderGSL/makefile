#!/bin/bash
# Get the location of this makefile.
# install realpath otherwise it won't work
ROOT_DIR := $(shell dirname "$(realpath $(lastword $(MAKEFILE_LIST)))")
CFLAGS = -Wall -g -fPIC -std=c99 
LD_FLAGS = -lTestSuite -lGraphSearch -lOpenCL -lm
OBJDIR = obj/
TESTOBJDIR = test_obj/
TEST_INCLUDES = /Test_headers/
MY_INCLUDES = /headers/# define user's include directory path
C_FILES := $(wildcard sources/*.c)
C_FILES_TEST := $(wildcard Test_sources/*.c)
OBJ_FILES := $(addprefix obj/,$(notdir $(C_FILES:.c=.o)))
OBJ_FILES_TEST := $(addprefix test_obj/,$(notdir $(C_FILES_TEST:.c=.o)))
INCLUDES := -I/usr/local/include -I/usr/include/CL

myprog: libGraphSearch.so libTestSuite.so
	LD_LIBRARY_PATH="$(ROOT_DIR)/bin/Debug/" gcc $(CFLAGS) -I"$(ROOT_DIR)$(TEST_INCLUDES)" -I"$(ROOT_DIR)$(MY_INCLUDES)" "$(ROOT_DIR)/main.c" -o myprog -L"$(ROOT_DIR)/bin/Debug" $(LD_FLAGS)

libTestSuite.so: $(OBJ_FILES_TEST)
	gcc -shared -o bin/Debug/libTestSuite.so $(TESTOBJDIR)*.o -L"$(ROOT_DIR)/bin/Debug" -lGraphSearch -lOpenCL -lm

test_obj/%.o: Test_sources/%.c
	gcc $(CFLAGS) -I"$(ROOT_DIR)$(TEST_INCLUDES)" -I"$(ROOT_DIR)$(MY_INCLUDES)" $(INCLUDES) -c -o "$(ROOT_DIR)/$@" "$(ROOT_DIR)/$<"

libGraphSearch.so: $(OBJ_FILES)
	gcc -shared -o bin/Debug/libGraphSearch.so $(OBJDIR)*.o -lOpenCL -lm

obj/%.o: sources/%.c
	gcc $(CFLAGS) -I"$(ROOT_DIR)$(MY_INCLUDES)" $(INCLUDES) -c -o "$(ROOT_DIR)/$@" "$(ROOT_DIR)/$<"

clean:
	rm $(TESTOBJDIR)*.o $(OBJDIR)*.o

