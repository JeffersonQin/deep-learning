#!/bin/bash
# This script is used to sync the files from the server to the local machine

MYDIR=`dirname $0`
cd $MYDIR
MYDIR=`pwd`
# Delete the old files
rm -rf $MYDIR/../d2l
# Relocate to working directory
cd $MYDIR/../
# Reset permissions
chmod 600 $MYDIR/gpu.pem
# Download the new files
sftp -b $MYDIR/sftp_commands.txt -i $MYDIR/gpu.pem ubuntu@gpu.sdl.moe
