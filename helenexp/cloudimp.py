#coding=gbk
"""
云端Import & 自动更新Python脚本
"""

import sys
import os
import urllib2
import md5
import urlparse
import socket
socket.setdefaulttimeout(1.0)
def cloudimport(url):
    fname = urlparse.urlsplit(url).path.split('/')[-1]
    hexpart = md5.md5(url).hexdigest()[:8]
    mname = fname.split('.')[-2]
    totalname = mname+'_'+hexpart

    try:
        doc = urllib2.urlopen(url).read()
    except:
        doc = None
    
    if doc!=None and ((not os.path.exists(totalname+'.py')) or
                      doc!=file(totalname+'.py','rb').read()):
        print "REFRESH MODULE",totalname
        file(totalname+'.py','wb').write(doc)
    
    module = __import__(totalname)
    sys.modules[mname] = module
    return module

