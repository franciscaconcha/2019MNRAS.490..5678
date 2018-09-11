.PHONY: exec lib debug setprob

# Set default problem setup
PROB = none

# Set default Anderson acceleration parameter
AA_M = 5

# Set default operation mode; set to TEST to enable testing mode
MODE = NORMAL

# Set the default for profiling to FALSE; note that profiling only
# works with GNU compilers at present
PROFILE = FALSE

# Uncomment this to enable the code to compile with GSL version 1.x;
# default is to use version 2.x
#GSLVERSION = 1

exec: setprob setprobmain
	@(cd vader_csrc; $(MAKE) exec AA_M=$(AA_M) MODE=$(MODE) PROFILE=$(PROFILE) GSLVERSION=$(GSLVERSION))
	@(if [ ! -e bin ]; \
	then \
		mkdir bin; \
	fi)
	@(cp vader_csrc/vader.ex bin)

lib: setprob
	@(cd vader_csrc; $(MAKE) lib AA_M=$(AA_M) MODE=$(MODE) PROFILE=$(PROFILE) GSLVERSION=$(GSLVERSION))
	@(if [ ! -e bin ]; \
	then \
		mkdir bin; \
	fi)
	@(cp vader_csrc/libvader.* bin)

debug: setprob setprobmain
	@(cd vader_csrc; $(MAKE) debug AA_M=$(AA_M) MODE=$(MODE) PROFILE=$(PROFILE) GSLVERSION=$(GSLVERSION))
	@(if [ ! -e bin ]; \
	then \
		mkdir bin; \
	fi)
	@(cp vader_csrc/vader.ex bin)

setprob:
	@echo "*** Setting problem to $(PROB) ***"
	@(if [ ! -e vader_csrc/userFunc.c ]; \
	then \
		cd vader_csrc/; cp -f prob/userFunc_$(PROB).c userFunc.c; touch userFunc.c; \
	fi)
	@(if ! cmp vader_csrc/userFunc.c vader_csrc/prob/userFunc_$(PROB).c ; \
	then \
		cd vader_csrc/; cp -f prob/userFunc_$(PROB).c userFunc.c; touch userFunc.c; \
	fi)

setprobmain:
	@(if [ ! -e vader_csrc/main.c ]; \
	then \
		cd vader_csrc/; cp -f prob/main_$(PROB).c main.c; touch main.c; \
	fi)
	@(if ! cmp vader_csrc/main.c vader_csrc/prob/main_$(PROB).c ; \
	then \
		cd vader_csrc/; cp -f prob/main_$(PROB).c main.c; touch main.c; \
	fi)

clean:
	rm -f bin/vader.ex bin/libvader.*
	@(cd vader_csrc; $(MAKE) clean)
