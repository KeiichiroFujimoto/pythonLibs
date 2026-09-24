from lxml import etree
import xmltodict

class LxmlHandler():

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def generateNewTree(rootElementName=None):
        root = etree.Element(rootElementName)
        tree = etree.ElementTree(root)
        return tree

    @staticmethod
    def getTreeFromXML(filePathXml=None,asDictionary=False):
        tree = etree.parse(open(filePathXml))
        if asDictionary:
            tree = LxmlHandler.getXmlAsDictionary(tree=tree, encoding='utf-8')
        return tree
    
    @staticmethod
    def readXmlAsDictionary(filePathXml=None):
        tree           = LxmlHandler.getTreeFromXML(filePathXml=filePathXml)
        dic_xml        = LxmlHandler.getXmlAsDictionary(tree=tree, encoding='utf-8')  
        del tree
        return dic_xml

    @staticmethod
    def getXmlAsString(tree=None, filePathXml=None, encoding='utf-8'):
        if tree is None:
            tree = LxmlHandler.getTreeFromXML(filePathXml=filePathXml)
        xmlStr = etree.tostring(tree, encoding=encoding)
        return xmlStr

    @staticmethod
    def getTreeFromXmlString(xmlStr=None):
        try:
            # Try parsing as bytes if encoding declaration is present
            if xmlStr.strip().startswith("<?xml"):
                xml_bytes = xmlStr.encode('ascii')
                root = etree.fromstring(xml_bytes)
            else:
                root = etree.fromstring(xmlStr)

            tree = etree.ElementTree(root)
            return tree

        except etree.XMLSyntaxError as e:
            print(f"XML Syntax Error: {e}")
        except ValueError as e:
            print(f"Value Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return None

    # @staticmethod
    # def getTreeFromXmlString(xmlStr=None):
    #     from lxml import etree
    #     root = etree.fromstring(xmlStr)
    #     tree = etree.ElementTree(root)
    #     return tree

    @staticmethod
    def getXmlAsDictionary(tree=None, encoding='utf-8'):
        xmlStr  = LxmlHandler.getXmlAsString(tree=tree, encoding=encoding)
        xmlDict = xmltodict.parse(xmlStr)
        return xmlDict
    
    @staticmethod
    def replace_numeric_elements(xml_string):
        import re
        # Pre-process the XML string to replace numeric element names
        def replace_numeric_tags(match):
            return f'<item{match.group(1)}>'

        # Replace opening tags
        xml_string = re.sub(r'<(\d+)>', replace_numeric_tags, xml_string)
        # Replace closing tags
        xml_string = re.sub(r'</(\d+)>', r'</item\1>', xml_string)

        # Parse the modified XML string as bytes
        root = etree.fromstring(xml_string.encode('utf-8'))
    
        # Function to recursively replace numeric element names
        def replace_elements(element):
            for child in element:
                replace_elements(child)
    
        # Replace numeric elements starting from the root
        replace_elements(root)

        return etree.tostring(root, pretty_print=True).decode()

    @staticmethod
    def getXmlFromDictionary(xmlDict=None):
        xmlStr    = xmltodict.unparse(xmlDict, pretty=True)
        print('xmlStr:'+str(xmlStr))
        xmlStr    = LxmlHandler.replace_numeric_elements(xml_string=xmlStr)
        xml_bytes = xmlStr.encode('utf-8')
        elem      = etree.fromstring(text=xml_bytes)
        tree      = etree.ElementTree(elem)
        return tree

    @staticmethod
    def getXmlElementFromXmlString(xmlString=None):
        xml_bytes = xmlString.encode('utf-8')
        elem      = etree.fromstring(text=xml_bytes)
        return elem

    @staticmethod
    def writeToXML(tree=None, filePathXml=None):
        with open(filePathXml, "wb") as f:
            tree.write(f, xml_declaration=True, pretty_print=True)

    @staticmethod
    def readXML(filePathXml=None):
        return LxmlHandler.getTreeFromXML(filePathXml=filePathXml)

    @staticmethod
    def getElementsByXpath(tree=None, xpath=None):
        elemList = tree.xpath(xpath)
        return elemList
    
    @staticmethod
    def getTextByXpath(tree=None, filePathXML=None, xpath=None):

        if tree is None and filePathXML is not None:
            tree = LxmlHandler.readXML(filePathXml=filePathXML)
        
        elemList = LxmlHandler.getElementsByXpath(tree=tree, xpath=xpath)
        for elem in elemList:
            return elem.text
        
        if filePathXML is not None:
            del tree

    @staticmethod
    def changeTextByXpath(tree=None, filePathXml=None, withOutputFile=None, xpath=None, text=None):
        
        if tree is None and filePathXml is not None:
            tree = LxmlHandler.readXML(filePathXml=filePathXml)

        index_           = None
        xpathBaseRec     = None
        xpathBaseRecList = []
        xpathBase        = ""
        for i, xpath_chk in enumerate(xpath.split('/')):
            if i != 0:
                xpathBase += "/" + xpath_chk

            if ':' in xpath_chk:
                xpathBaseRec = xpathBase
                index_       = int(xpath_chk.split(':')[1])
                xpathBaseRecList.append(xpathBaseRec.replace(':'+str(index_),''))
                xpathBase = ""
        
        xpathBaseRecList.append(xpathBase)
        
        if index_ != None:
            elemList     = LxmlHandler.getElementsByXpath(tree=tree, xpath=xpathBaseRecList[0])
            for i, elem in enumerate(elemList):
                if index_ == i:
                    elemList2 = LxmlHandler.getElementsByXpath(tree=elem, xpath="."+xpathBaseRecList[1])
                    for j, elem2 in enumerate(elemList2):
                        print('>>> elem2:'+str(elem2))
                        elem2.text = text

        if index_ == None:
            elemList = LxmlHandler.getElementsByXpath(tree=tree, xpath=xpath)
            for i, elem in enumerate(elemList):
                print('>>> elem:'+str(elem))
                elem.text = text

        if withOutputFile and filePathXml is not None:
            LxmlHandler.writeToXML(tree=tree, filePathXml=filePathXml)

    @staticmethod
    def changeTextByXpathList(tree=None, filePathXML=None, xpathList=None, textList=None):

        if tree is None and filePathXML is not None:
            tree = LxmlHandler.readXML(filePathXml=filePathXML)

        for ii in range(len(xpathList)):
            xpath = xpathList[ii]
            text  = textList[ii]
            LxmlHandler.changeTextByXpath(tree=tree,xpath=xpath,text=text)
        
        if filePathXML is not None:
            LxmlHandler.writeToXML(tree=tree, filePathXml=filePathXML)
            del tree
            tree = None

    @staticmethod
    def getAttirubuteValue(elem=None,attributeName=None):
        return elem.get(attributeName)

    @staticmethod
    def getAttirubuteValueByXpath(filePathXML=None, tree=None, xpath=None, attributeName=None, returnWithElementList=False):
        try:
            print('##### filePathXML:'+str(filePathXML))
            import os
            print('Current dir:'+str(os.getcwd()))
            treeIsGiven = tree is not None
            if tree is None and filePathXML is not None:
                tree = LxmlHandler.readXML(filePathXml=filePathXML)
            
            elemList = LxmlHandler.getElementsByXpath(tree=tree, xpath=xpath)

            attributeValueList = []
            for elem in elemList:
                attributeValueList.append(LxmlHandler.getAttirubuteValue(elem=elem,attributeName=attributeName))
            
            if returnWithElementList:
                return attributeValueList, elemList, tree
            else:
                if treeIsGiven == False:
                    del tree
                return attributeValueList

        except Exception as e:
            return f"Error: {e}"
    
    @staticmethod
    def changeAttributeValueByXpath(tree=None, xpath=None, attributeName=None, attributeValue=None):
        elemList = LxmlHandler.getElementsByXpath(tree=tree, xpath=xpath)
        for elem in elemList:
            print('>>> elem:'+str(elem))
            LxmlHandler.changeAttributeValue(elem=elem,attributeName=attributeName,attributeValue=attributeValue)

    @staticmethod
    def changeAttributeValue(elem=None, attributeName=None, attributeValue=None):
        elem.set(attributeName, attributeValue)

    @staticmethod
    def changeAttributeValueByXpathList(filePathXML=None, tree=None, xpathList=None, attributeNameList=None, attributeValueList=None):
        try:
            if tree is None and filePathXML is not None:
                tree = LxmlHandler.readXML(filePathXml=filePathXML)

            for ii in range(len(xpathList)):
                xpath          = xpathList[ii]
                attributeValue = attributeValueList[ii]
                attributeName  = attributeNameList[ii]
                LxmlHandler.changeAttributeValueByXpath(tree=tree,xpath=xpath,attributeName=attributeName,attributeValue=attributeValue)

            if filePathXML is not None:
                LxmlHandler.writeToXML(tree=tree, filePathXml=filePathXML)
                del tree
                tree = None

        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def changeAttributeValueByXpathListWithKeyCheck(filePathXML=None,xpathList=None,attributeNameList=None,attributeValueList=None,
                                                    attributeNameKey=None,attributeValueKey=None):

        for ii in range(len(xpathList)):
            xpath = xpathList[ii]
            attributeName  = attributeNameList[ii]
            attributeValue = attributeValueList[ii]
            print('###### xpath:'+str(xpath))
            attributeValueKeyList, elemList, tree = LxmlHandler.getAttirubuteValueByXpath(filePathXML=filePathXML, 
                                                                                        xpath=xpath, 
                                                                                        attributeName=attributeNameKey, 
                                                                                        returnWithElementList=True)

            for ii in range(len(attributeValueKeyList)):
                if attributeValueKeyList[ii] == attributeValueKey:
                    elem = elemList[ii]
                    LxmlHandler.changeAttributeValue(elem=elem,attributeName=attributeName,attributeValue=attributeValue)

        LxmlHandler.writeToXML(tree=tree, filePathXml=filePathXML)
        del tree
        tree = None

    @staticmethod
    def getParent(elem):
        return elem.getparent()

    @staticmethod
    def removeFromParent(parent,child):
        parent.remove(child)
        
    @staticmethod
    def getNewElement(name=""):
        return etree.Element(name)

    @staticmethod
    def insertChild(parent=None,child=None,indexChild=None):
        parent.insert(indexChild,child)
        
    @staticmethod
    def appendChild(parent=None,child=None):
        parent.append(child)
    
    @staticmethod
    def insertElementBetweenParentAndChild(child=None,name_elemNew="",indexChild=None):
        parent = LxmlHandler.getParent(elem=child)
        LxmlHandler.removeFromParent(parent=parent,child=child)
        elemNew = LxmlHandler.getNewElement(name=name_elemNew)
        LxmlHandler.insertChild(parent=parent,child=elemNew,indexChild=indexChild)
        LxmlHandler.appendChild(parent=elemNew,child=child)
        elemNew.append(child)

    @staticmethod
    def getNumberOfChildren(parent=None):
        numOfChilds = len(parent.getchildren())
        return numOfChilds

    @staticmethod
    def createNewElement(name_elemNew=None, text_elemNew=None):
        element = etree.Element(name_elemNew)
        element.text = text_elemNew
        return element
    
    @staticmethod
    def createNewElementByXpath(tree=None, xpath=None, name_elemNew=None, text_elemNew=None):
        elems = LxmlHandler.getElementsByXpath(tree=tree, xpath=xpath)
        for elem in elems:
            elemNew = etree.SubElement(elem, name_elemNew)
            if text_elemNew is not None:
                elemNew.text = text_elemNew

    @staticmethod
    def getValuesBySplit(value=None,separator=None,type=float):
        valueListStr = value.split(' ') 
        valueList    = []
        for ii in range(len(valueListStr)):
            if type==float:
                valueList.append(float(valueListStr[ii]))
        return valueList

    @staticmethod
    def findElementList(root=None, tagName=None, tagNameChild=None, elemChildValue=None):
        elemList = root.findall(tagName)
        elemListFound = []
        for elem in elemList:
            id_element = elem.find(tagNameChild)
            if id_element is not None and id_element.text == elemChildValue:
                elemListFound.append(elem)
        return elemListFound

    @staticmethod
    def getChildElementListByTagName(elemParent=None, tagName=None):
        return elemParent.findall(tagName)
    
    @staticmethod
    def getXmlTree(tree=None, filePathXml=None):
        if tree is None:
            tree = LxmlHandler.getTreeFromXML(filePathXml=filePathXml)
        else:
            return tree
        return tree
    
    @staticmethod
    def display(tree=None):
        from lxml import etree
        print(etree.tostring(tree, pretty_print=True).decode())

if __name__ == '__main__':

    tree = LxmlHandler.getTreeFromXML(filePathXml="./input.xml")
    
    for i, s in enumerate(tree.xpath("//s")):
        LxmlHandler.insertElementBetweenParentAndChild(child=s,name_elemNew="c",indexChild=0)
        break

    LxmlHandler.writeToXML(tree=tree, filePathXml="./output.xml")

