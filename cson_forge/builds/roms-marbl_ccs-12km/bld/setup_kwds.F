#include "cppdefs.opt"

      subroutine setup_kwds (ierr)

!!!!!! WARNING: THIS IS A MACHINE GENERATED CODE, DO NOT EDIT! !!!!!!
!!!!!! This file needs to be updated only if new keywords were !!!!!!
!!!!!! introduced into "read_inp.F". To create or refresh this !!!!!!
!!!!!! file use compile and execute "checkkwds.F" as an        !!!!!!
!!!!!! independent program, or use commands "make checkkwds"   !!!!!!
!!!!!! or "make depend".                                       !!!!!!

      use param
      use scalars
      use strings
      implicit none
      integer ierr, is,ie
      do is=1,max_opt_size
        kwds(is:is)=' '
      enddo
      is=1
#ifdef MARBL
#ifdef MARBL_DIAGS
#endif
#endif
#ifdef SOLVE3D
#endif
#ifdef INT_IARGC
#endif
#ifdef SOLVE3D
# ifdef LMD_KPP
# endif
#endif
#ifdef MPI
#endif
#ifdef MPI
#endif
      ie=is + 5
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='title'
      kwds(ie+1:ie+1)=' '
      is=ie+2
      ie=is +13
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='time_stepping'
      kwds(ie+1:ie+1)=' '
      is=ie+2
#ifndef SOLVE3D
#endif
#ifdef MARBL
      ie=is +21
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='MARBL_biogeochemistry'
      kwds(ie+1:ie+1)=' '
      is=ie+2
#ifdef MARBL_DIAGS
#endif
#endif
#ifdef SOLVE3D
      ie=is + 7
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='S-coord'
      kwds(ie+1:ie+1)=' '
      is=ie+2
      ie=is + 4
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='rho0'
      kwds(ie+1:ie+1)=' '
      is=ie+2
# ifndef NONLIN_EOS
      ie=is +11
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='lin_rho_eos'
      kwds(ie+1:ie+1)=' '
      is=ie+2
#  ifdef SALINITY
#  endif
#  ifdef SALINITY
#  endif
# endif
#endif /* SOLVE3D */
#ifdef UV_VIS2
      ie=is +12
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='lateral_visc'
      kwds(ie+1:ie+1)=' '
      is=ie+2
# if !defined EW_PERIODIC || !defined NS_PERIODIC
      ie=is + 6
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='gamma2'
      kwds(ie+1:ie+1)=' '
      is=ie+2
# endif
#endif
#ifdef SOLVE3D
# ifdef TS_DIF2
      ie=is +12
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='tracer_diff2'
      kwds(ie+1:ie+1)=' '
      is=ie+2
# endif
#endif
      ie=is +11
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='bottom_drag'
      kwds(ie+1:ie+1)=' '
      is=ie+2
#ifdef SOLVE3D
#endif
#ifdef SOLVE3D
#endif
#ifdef SOLVE3D
# if !defined LMD_MIXING && !defined BVF_MIXING && !defined PP_MIXING\
                         && !defined MY2_MIXING && !defined MY25_MIXING
      ie=is +15
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='vertical_mixing'
      kwds(ie+1:ie+1)=' '
      is=ie+2
# endif
# ifdef MY25_MIXING
      ie=is +13
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='MY_bak_mixing'
      kwds(ie+1:ie+1)=' '
      is=ie+2
#  ifdef Q_DIF2
#  endif
#  ifdef Q_DIF4
#  endif
# endif
# if defined SFLX_CORR && defined SALINITY
      ie=is +14
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='SSS_correction'
      kwds(ie+1:ie+1)=' '
      is=ie+2
# endif
# if defined QCORRECTION
      ie=is +14
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='SST_correction'
      kwds(ie+1:ie+1)=' '
      is=ie+2
# endif
#endif /* SOLVE3D */
#if  defined T_FRC_BRY || defined M2_FRC_BRY || defined TNUDGING \
  || defined Z_FRC_BRY || defined M3_FRC_BRY || defined M2NUDGING \
                       || defined M3NUDGING  || defined WKB_FRC_BRY
      ie=is + 5
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='ubind'
      kwds(ie+1:ie+1)=' '
      is=ie+2
# ifdef SOLVE3D
# endif
# ifdef SOLVE3D
# endif
# ifdef SOLVE3D
# endif
#endif
#ifdef SPONGE
      ie=is + 8
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='v_sponge'
      kwds(ie+1:ie+1)=' '
      is=ie+2
#endif
#ifndef ANA_GRID
      ie=is + 4
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='grid'
      kwds(ie+1:ie+1)=' '
      is=ie+2
# if defined MPI && defined PARALLEL_FILES
# endif
#endif
      ie=is + 7
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='initial'
      kwds(ie+1:ie+1)=' '
      is=ie+2
#ifdef ANA_INITIAL
#endif
#if defined MPI && defined PARALLEL_FILES
#endif
#ifdef ANA_INITIAL
#endif
      ie=is + 7
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='forcing'
      kwds(ie+1:ie+1)=' '
      is=ie+2
# if defined MPI && defined PARALLEL_FILES
# endif
# ifdef MPI_SILENT_MODE
# endif
# ifdef MPI_SILENT_MODE
# endif
#if defined SOLVE3D &&( defined TCLIMATOLOGY ||( defined TNUDGING && \
                        defined T_FRC_BRY )) && !defined ANA_TCLIMA
      ie=is +11
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='climatology'
      kwds(ie+1:ie+1)=' '
      is=ie+2
# if defined MPI && defined PARALLEL_FILES
# endif
#endif
      ie=is +16
      if (ie>=max_opt_size) goto 99
      kwds(is:ie)='output_root_name'
      kwds(ie+1:ie+1)=' '
      is=ie+2
#ifdef MPI
#endif
      return
  99  mpi_master_only write(*,'(/1x,2A/25x,A/)')
     &  '### ERROR: setup_kwds :: Insufficient size of str "kwds" ',
     &  'in file "strings".',  'Increase the size it and recompile.'
      ierr=ierr+1
      return
      end
