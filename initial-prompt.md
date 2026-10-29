# Initial prompt: add tensor parallelism
I want to add tensor parallelism to the engine to run a two tensor parallel job with the GLM 5.3 Flash model Maya-S. I want to run on two twin sysmtems strix-halo-1.local and strix-halo-2.local (this one) who have the same environment installed and are both AMD gfx1151 platforms with CachyOS. I have already added tensor parallelism to another platform in the project /home/ugo/software/gufo which you can use as a reference which also includes information about the network connection with RDMA over two parallel thunderbolt 4 cables.
1. Perform an in-depth analysis to understand what's needed and record the finding in a document.
2. Use the outcome of the analysis to design a solution an record the design in a document.
3. Use the analysis and design information to write a detailed implementation plan
DO NOT WRITE ANY CODE NOR RUN ANY PROCESS.
YOU CANNOT RUN ANY PROCESS BECAUSE I AM RUNNING A BENCHMARK ON BOTH SYSTEMS
