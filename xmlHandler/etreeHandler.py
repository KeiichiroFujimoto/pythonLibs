import xml.etree.ElementTree as ET

class etreeHandler():
    
    root      = None
    parentMap = None

    def openXml(self, fileNameXmlIn=None):
        tree      = ET.parse(fileNameXmlIn)
        self.root = tree.getroot()
        etreeHandler.addParentInfo(elem=self.root)
        self.parentMap = etreeHandler.getParentMap(self.root)

    @staticmethod
    def getElementDepth(elem):
        d = 0
        while elem is not None:
            d += 1
            elem = etreeHandler.getParent(elem=elem)
        return d
    
    @staticmethod
    def addParentInfo(elem):
        for child in elem:
            child.attrib['__my_parent__'] = elem
            etreeHandler.addParentInfo(child)

    @staticmethod
    def stripParentInfo(elem):
        for child in elem:
            child.attrib.pop('__my_parent__', 'None')
            etreeHandler.stripParentInfo(child)

    @staticmethod
    def getParent(elem):
        if '__my_parent__' in elem.attrib:
            return elem.attrib['__my_parent__']
        else:
            return None
    
    @staticmethod
    def parseAll(root=None,elemKey=None):
        for element in root.iter(elemKey):
            for elmCld in element.iter():
                depth = etreeHandler.getElementDepth(elmCld)
                print("depth:"+str(etreeHandler.getElementDepth(elmCld)))
                print("elemCld:"+str(elmCld.tag))
                print("elemCld:"+str(elmCld.text))

    @staticmethod
    def getParentMap(root):
        return {c:p for p in root.iter() for c in p}

    @staticmethod
    def getElementsByPath(root=None, path=None):
        elements = root.findall(path)
        return elements

    def getPath(self, elem=None, root=None, withRoot=False):
        arr = []
        while elem != root:
            arr.append(elem.tag)
            elem = self.parentMap[elem]
        if withRoot:
            arr.append(root.tag)
        path = '/'.join(arr[::-1]) 
        return path

    def extractTextInfo(self, root, original_root):

        for child in root:
            if child.text is not None and len(child.text.strip()) > 0:
                c = child
                arr = []
                while c != original_root:
                    arr.append(c.tag)
                    c = self.parentMap[c]
                arr.append(original_root.tag)

                print('/'.join(arr[::-1]))
                print(child.text)

            etreeHandler.extractTextInfo(child, original_root)
