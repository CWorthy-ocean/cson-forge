#include "cppdefs.opt"

      subroutine check_srcs

!!!!!! WARNING: THIS IS A MACHINE GENERATED CODE, DO NOT EDIT! !!!!!!
!!!!!! This file needs to be updated only if the new files     !!!!!!
!!!!!! were introduced into or deleted from the list of source !!!!!!
!!!!!! codes SRCS in the Makefile.                             !!!!!!


      use strings
      implicit none
      integer i
      do i=1,max_opt_size
        srcs(i:i)=' '
      enddo
      srcs(   1:   4)='SRCS'
      srcs(   5:  15)=' $(wildcard'
      srcs(  16:  30)=' *$(UPF77_ext))'
      srcs(  31:  35)=' SRCS'
      srcs(  36:  37)=' :'
      srcs(  38:  50)=' $(filter-out'
      srcs(  51:  59)=' $(EXCL),'
      srcs(  60:  68)=' $(SRCS))'
      srcs(  69:  73)=' SRCS'
      srcs(  74:  75)=' :'
      srcs(  76:  83)=' $(SRCS)'
      srcs(  84:  91)=' $(INCL)'
      return
      end
