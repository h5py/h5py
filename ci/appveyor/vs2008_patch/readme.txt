How to enable X64 and IA64 programming in Visual C++ Express:

1. Install Visual C++ 2008 Express (to default folder in C drive, or this patch will not work)
2. Install Windows SDK (Microsoft Windows SDK for Windows 7 and .NET Framework 3.5 SP1)
3. Fix SDK bugs: http://www.cppblog.com/xcpp/archive/2009/09/09/vc2008express_64bit_win7sdk.html
4. Open a command prompt with Administrator privilege, navigate to the folder contains this file, run setup_x86.bat or setup_x64.bat according to your OS architecture
5. If there is no error in the command prompt, launch the Visual C++ 2008 Express IDE and build your X64 or IA64 projects

This work is based on the work by jenshuebel: http://jenshuebel.wordpress.com/2009/02/12/visual-c-2008-express-edition-and-64-bit-targets/

Thanks jenshuebel for the complete and accurate instructions, and thanks Microsoft for the free Visual C++ IDE.



Xia Wei, sunmast#gmail.com
