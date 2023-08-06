## Testing push to repo

## HLS C simulation
If you experience CSIM error as:
ERROR: [SIM 211-100] 'csim_design' failed: compilation error(s).

It's due to stack size not big enough
Simply increase the stack size for compilation by setting
Project Settings-->Simulation-->Linker Flags
as
-Wl,--stack,10485670


## C synthesis
make sure the part number is: xqzu5ev-ffrb900-1-i


## Ubuntu 20.04 username and password
username: ubuntu
password: comp4601


## Serial port (CMD) via Putty
to access Ubuntu CMD interface via serial port, make sure the USB is connected between host PC and Kria
then use Putty to connect to COM port (serial, COM4, speed 115200)

## SFTP via WinScp
Login with the eth0 ip address assigned to Kria board and Ubuntu username and password 


## Running Pynq Python script
Get into sudo mode using: sudo su
First source the Python venv using: source /etc/profile.d/pynq_venv.sh
then run SkyNet.py

