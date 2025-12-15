      subroutine add_git_hash(ncid)

      use roms_read_write
      use nc_read_write
      use netcdf

      implicit none
      integer :: ncid,ierr

      git_hash="64d168e1405582b24f639dadbb180d071035b6ac"

      ierr=nf90_put_att(ncid, nf90_global, 'git_version',git_hash)
      if (ierr/=nf90_noerr)
     &    call handle_ierr(ierr,'add_git hash: ')


      end subroutine add_git_hash

