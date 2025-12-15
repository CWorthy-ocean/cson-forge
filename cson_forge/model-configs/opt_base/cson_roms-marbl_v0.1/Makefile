# Just type: make
# (In an example directory or work directory)

# This makefile does:
# 1) Makes a sync of the distribution code to a Compile
#    directory here (will create directory if not here yet)
# 2) Copies local files in this directory into that Compile directory
# 3) Compiles code into executable for the example.
# 4) Copies 'roms' from Compile dir into this dir.

# This line de-activates all implicit rules
.SUFFIXES:

.PHONY: all clean depend

all:
	rsync -a $(ROMS_ROOT)/src/*.F $(ROMS_ROOT)/src/*.h  $(BUILD_DIR)
	make show_tag
	@rsync -a $(ROMS_ROOT)/src/*.opt $(BUILD_DIR)
	@rsync -a $(ROMS_ROOT)/src/Makedefs.inc $(BUILD_DIR)
	@rsync -a $(ROMS_ROOT)/src/Make.depend $(BUILD_DIR)
	@rsync -a $(ROMS_ROOT)/src/Makefile $(BUILD_DIR)
	@rm $(BUILD_DIR)/*.f	2>/dev/null || :
	cp -p *.h *.F *.opt Makedefs.inc $(BUILD_DIR) 2>/dev/null || :
	cd $(BUILD_DIR); make depend 2>/dev/null || :
	cd $(BUILD_DIR); make -j6;


# note: "2>/dev/null || :" above is needed to suppress error
#       stop if no .h or .F files in current directory.
#       Also, makedepf90 in make depend can throw out strange
#       errors for .h files for inexplicable changes made to
#       cppdefs.opt.

# for adding git hash:
tag := $(shell git rev-parse HEAD)
show_tag:
	@perl -i -pe 's/git_hash=.*/git_hash="$(tag)"/' $(BUILD_DIR)/add_git_hash.*

compile_clean:
	rm -r $(BUILD_DIR)/ exe/roms.$(GRID_NAME) 2>/dev/null || :

work_clean:
	rm -r $(BUILD_DIR)/ *.F *.h *.in *.sh *.nc exe/roms.$(GRID_NAME) 2>/dev/null || :

code_check_clean:
	@cd $(ROMS_ROOT)/Examples/Flux_frc        ; make compile_clean
	@cd $(ROMS_ROOT)/Examples/Pipes_ana       ; make compile_clean
	@cd $(ROMS_ROOT)/Examples/Pipes_real      ; make compile_clean
	@cd $(ROMS_ROOT)/Examples/Rivers_ana      ; make compile_clean
	@cd $(ROMS_ROOT)/Examples/Rivers_real     ; make compile_clean
	@cd $(ROMS_ROOT)/Examples/Tracers_passive ; make compile_clean
	@cd $(ROMS_ROOT)/Examples/WEC_real        ; make compile_clean


copy_to_Work_dir:
	cp -p *.F *.h *.in *.sh ../../Work 2>/dev/null || :

nhmg:
	cd $(ROMS_ROOT)/NHMG/src; make clean; make

tools-roms:
	cd $(ROMS_ROOT)/Tools-Roms/; make ;


# The no-print suppress makes messages of entering/leaving directories
MAKEFLAGS += --no-print-directory
