

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