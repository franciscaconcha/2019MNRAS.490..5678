# Specify compiler if desired
#CC		= gcc

# If we need to add the GSL to the include and load library paths, do it here
#C_INCLUDE_PATH  += PATH_TO_GSL_HEADERS
#LD_LIBRARY_PATH  += PATH_TO_GSL_OBJECTS

# Set default Anderson acceleration parmeter
AA_M		= 5
DEFINES		+= -DAA_M=$(AA_M)

# Compile in testing mode if requested
MODE = NORMAL
ifeq ($(MODE), TEST)
	DEFINES += -DTESTING_MODE
endif

# Turn on profiling if desired
ifeq ($(PROFILE), gprof)
	CPROF  = -pg
	LDPROF = -pg
endif
ifeq ($(PROFILE), gperftools)
	CPROF  = -Wl,-no_pie
	LDPROF = -L/Users/krumholz/Library/lib/lib -lprofiler
endif

# Compile in GSL version 1 mode if requested
GSLVERSION = 2
ifeq ($(GSLVERSION), 1)
	DEFINES += -DGSL1
endif

# Set flags for debug and non-debug mode
COPTFLAGS	= -O3 -MMD -MP -I $(C_INCLUDE_PATH) -DHAVE_INLINE $(DEFINES) -Wall $(CPROF)
LDOPTFLAGS	= -O3 -lgsl -lgslcblas -lm $(LDPROF)
CDEBFLAGS	= -g -MMD -MP -I $(C_INCLUDE_PATH) $(DEFINES) -Wall $(CPROF)
LDDEBFLAGS	= -g -lgsl -lgslcblas -lm $(LDPROF)

# Add library flags if supplied
ifdef LD_LIBRARY_PATH
    LDOPTFLAGS += -L $(LD_LIBRARY_PATH)
    LDDEBLAGS += -L $(LD_LIBRARY_PATH)
endif
ifdef LIBRARY_PATH
    LDOPTFLAGS += -L $(LIBRARY_PATH)
    LDDEBLAGS += -L $(LIBRARY_PATH)
endif

# Set default compilation flags to non-debug ones; can be overridden below
CFLAGS	= $(COPTFLAGS)
LDFLAGS	= $(LDOPTFLAGS)

# Name for standalone executable output
EXENAME		= vader.ex

# Name and compilation flags for dynamically linked library mode
OS 		:= $(shell uname)
ifeq ($(OS), Darwin)
	LIBNAME	   = libvader.dylib
	CLIBFLAGS  =
	DYNLIBFLAG = -dynamiclib
else ifeq ($(OS), Linux)
	LIBNAME    = libvader.so
	CLIBFLAGS  = -fPIC
	DYNLIBFLAG = -shared
endif

# Default target
.PHONY: debug exec lib
lib:	CFLAGS += $(CLIBFLAGS)
lib:	$(LIBNAME)

# Include dependencies
-include $(DEPS)

# Set flags for debug mode
debug: CFLAGS	= $(CDEBFLAGS)
debug: LDFLAGS	= $(LDDEBFLAGS)
debug: exec

# Pointers to sources
SOURCES		= $(wildcard *.c)
OBJECTS		= $(SOURCES:%.c=%.o)
DEPS		= $(SOURCES:%.c=%.d)

exec:	$(EXENAME)

$(LIBNAME):	$(OBJECTS) userFunc.o
	$(CC) $(LDFLAGS) $(DYNLIBFLAG) -o $(LIBNAME) $^

$(EXENAME):	$(OBJECTS) userFunc.o
	$(CC) -o $(EXENAME) $^ $(LDFLAGS)

compile:	$(OBJECTS)

userFunc.o:	userFunc.c

userFunc.c:
	cp prob/userFunc_noOp.c userFunc.c

clean:
	rm -f $(LIBNAME) $(EXENAME) $(OBJECTS) $(DEPS)
	rm -f userFunc.c main.c
