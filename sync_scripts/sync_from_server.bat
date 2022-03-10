:: This script is used to sync the files from the server to the local machine
@echo off
:: Delete the old files
del /S /Q %~dp0\..\d2l

:: Reset permissions
	:: Set Key File Variable:
	Set Key="%~dp0\gpu.pem"

	:: Remove Inheritance:
	Icacls %Key% /c /t /Inheritance:d

	:: Set Ownership to Owner:
		:: Key's within %UserProfile%:
		Icacls %Key% /c /t /Grant %UserName%:F

		:: Key's outside of %UserProfile%:
		TakeOwn /F %Key%
		Icacls %Key% /c /t /Grant:r %UserName%:F

	:: Remove All Users, except for Owner:
	Icacls %Key% /c /t /Remove:g "Authenticated Users" BUILTIN\Administrators BUILTIN Everyone System Users

	:: Verify:
	Icacls %Key%

	:: Remove Variable:
	set "Key="

:: Relocate to working directory
cd %~dp0\..
:: Download the new files
sftp -b %~dp0\sftp_commands.txt -i %~dp0\gpu.pem ubuntu@gpu.sdl.moe
