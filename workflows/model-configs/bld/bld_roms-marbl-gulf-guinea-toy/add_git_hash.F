      subroutine add_git_hash(ncid)

      use roms_read_write
      use nc_read_write
      use netcdf

      implicit none
      integer :: ncid,ierr

      git_hash="1edf7ca56b1a9beca4f9c5df23909a2bc5689a26"

      ierr=nf90_put_att(ncid, nf90_global, 'git_version',git_hash)
      if (ierr/=nf90_noerr)
     &    call handle_ierr(ierr,'add_git hash: ')


      end subroutine add_git_hash

