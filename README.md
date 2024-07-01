# **Random Points / Troubleshooting**

All the python was written on a Mac OS X with an M3 Apple chip. I tried to make everything agnostic but if there are any problems it could be down to:

* For pytorch I used device "mps" instead of "cuda" to access the GPU.