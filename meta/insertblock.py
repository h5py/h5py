import os

python_mode = ('.py', '.pyx', '.pxd', '.pxi')
c_mode = ('.c','.h')

firstline = { 'python': "#+",
              'c': "/*"+"*"*4 + " Preamble block " + "*"*57}

lastline = { 'python': "#-",
             'c': '*'*6 + " End preamble block " + '*'*52 +'/' }

beginline = { 'python': '# ', 'c': '* ' }

def guessmode(filename):
    ext = os.path.splitext(filename)[1]
    if ext in python_mode:
        return 'python'
    if ext in c_mode:
        return 'c'
    raise ValueError("Can't determine mode of %s" % filename)

def iterblock(file_obj, mode, output_block=True):

    in_block = False

    for idx, line in enumerate(file_obj):
        matchline = line.strip()

        if matchline == firstline[mode]:
            in_block = True
            continue
        elif matchline == lastline[mode]:
            in_block = False
            continue

        if in_block == output_block:
            yield line

def printblock(filename):
    fh = open(filename,'r')
    try:
        print "".join(iterblock(fh, guessmode(filename), True))
    finally:
        fh.close()
    
def replaceblock(filename, block_filename):
    mode = guessmode(filename)
    fh = open(filename,'r')

    contents = list(iterblock(fh, mode, False))
    fh.close()

    fh_block = open(block_filename, 'r')
    block_contents = list(fh_block)
    fh_block.close()

    contents =  [firstline[mode]+os.linesep] + \
                [beginline[mode] + line for line in block_contents] + \
                [lastline[mode]+os.linesep] + \
                contents

    fh = open(filename,'w')
    fh.write("".join(contents))
    fh.close()

def eraseblock(filename):
    try:
        fh = open(filename,'r')
    except IOError:
        return
    try:
        contents = list(iterblock(fh, guessmode(filename),False))
    finally:
        fh.close()

    fh = open(filename, 'w')
    fh.write("".join(contents))
    fh.close()



