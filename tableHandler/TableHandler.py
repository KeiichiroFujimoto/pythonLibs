import sys
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.utils import get_column_letter
from pythonLibs.fileHandler import *
import re

class TableHandler():

    def __init__(self):
        print("This is constructor of ConvertTable")

    @staticmethod
    def getInitializedNumpyTable(numColumn, numRow, colNames=None):
        if colNames == None:
            tab_np = np.zeros(numRow, dtype=[('V' + str(icol), 'f8') for icol in range(numColumn)])
        else:
            tab_np = np.zeros(numRow, dtype=[(colNames[icol], 'f8') for icol in range(numColumn)])
        return tab_np

    @staticmethod
    def getInitializedPandasTable(numColumn, numRow, colNames=None):
        if colNames == None:
            tab_np = np.zeros(numRow, dtype=[('V' + str(icol), 'f8') for icol in range(numColumn)])
        else:
            tab_np = np.zeros(numRow, dtype=[(colNames[icol], 'f8') for icol in range(numColumn)])
        return pd.DataFrame(tab_np)

    @staticmethod
    def getNumpyTable(tab=None):
        if type(tab) is np.ndarray:
            tab_np = tab
        elif type(tab) is pd.core.frame.DataFrame:
            tab_np = np.zeros(tab.shape[0], dtype=[(tab.columns[i], 'f8') for i in range(tab.shape[1])])
            for irow in range(tab.shape[0]):
                for icol in range(tab.shape[1]):
                    tab_np[irow][icol] = tab.values[irow][icol]
        else:
            tab_np = None
        return tab_np

    @staticmethod
    def getPandasTable(tab=None):
        if type(tab) is np.ndarray:
            tab_pd = pd.DataFrame(tab)
        elif type(tab) is pd.core.frame.DataFrame:
            tab_pd = tab
        else:
            tab_pd = None
        return tab_pd

    @staticmethod
    def getColumnNameList(tab=None):
        if type(tab) is np.ndarray:
            columNameList = list(tab.dtype.names)
        else:
            columNameList = list(tab.columns)
        return columNameList

    @staticmethod
    def getColumnDataByNames(tab=None, colNames=None):
        if type(tab) is np.ndarray:
            colData = tab[colNames]
        elif type(tab) is pd.core.frame.DataFrame:
            colData = tab[colNames]
        else:
            colData = None
        return colData

    @staticmethod
    def getColumnDataSingleByIndex(tab=None, icol=None):
        colData = TableHandler.getColumnDataByIndex(tab=tab, icolStart=icol, icolEnd=icol)
        return colData

    @staticmethod
    def getColumnDataByIndex(tab=None, icolStart=None, icolEnd=None):
        if type(tab) is np.ndarray:
            colData = tab[list(tab.dtype.names[icolStart:(icolEnd + 1)])]
        else:
            colData = tab[tab.columns[icolStart:(icolEnd + 1)]]
        return colData

    @staticmethod
    def getSubarray(tab=None, icolStart=None, icolEnd=None, irowStart=None, irowEnd=None):
        if type(tab) is np.ndarray:
            subArray = tab[list(tab.dtype.names[icolStart:(icolEnd + 1)])][irowStart:(irowEnd + 1)]
        else:
            subArray = None
        return subArray

    @staticmethod
    def getColumnNumber(tab=None):
        if type(tab) is np.ndarray:
            return tab.shape[1]
        elif type(tab) is pd.core.frame.DataFrame:
            return len(tab.columns)

    @staticmethod
    def getRowNumber(tab=None):
        if type(tab) is np.ndarray:
            return tab.shape[0]
        elif type(tab) is pd.core.frame.DataFrame:
            return len(tab)

    @staticmethod
    def SaveAsExcel_TableSet(tableSet, filePath=None, sheetNameList=None):
        for i in range(len(tableSet)):
            tab = tableSet[i]
            df = TableHandler.getPandasTable(tab)
            if i == 0:
                with pd.ExcelWriter(filePath, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name=sheetNameList[i], index=False)
            else:
                with pd.ExcelWriter(filePath, engine="openpyxl", mode="a") as writer:
                    df.to_excel(writer, sheet_name=sheetNameList[i], index=False)

    @staticmethod
    def SaveAsExcel(tab, filePath=None, sheetName=None, withPlot=False, indexX=1, indexYList=None, colorList=None):
        if type(tab) is np.ndarray:
            tab_pd = pd.DataFrame(tab)
        elif type(tab) is pd.core.frame.DataFrame:
            tab_pd = tab

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = sheetName

        stcol = 1
        strow = 1
        for c in range(0, len(tab_pd.columns)):
            ws[get_column_letter(c + stcol) + str(strow)].value = tab_pd.columns[c]

        numColumn = TableHandler.getColumnNumber(tab)
        numRow = TableHandler.getRowNumber(tab)
        for r in range(0, numRow):
            for c in range(0, numColumn):
                ws[get_column_letter(c + stcol) + str(strow + r + 1)].value = tab_pd.iloc[r][c]

        if withPlot == True:
            from openpyxl.chart import ScatterChart, Reference, Series
            from openpyxl.drawing.colors import ColorChoice
            from openpyxl.styles import colors

            chart = ScatterChart()
            len_data = len(tab_pd)
            num_series = len(indexYList)
            min_row = 2
            max_row = min_row + len_data - 1
            x_values = Reference(ws, min_col=indexX, min_row=min_row, max_row=max_row)

            for i in range(0, num_series):
                min_col = indexYList[i]
                values = Reference(ws, min_col=min_col, min_row=min_row, max_row=max_row)
                con = Series(values, x_values, title=tab_pd.columns.values[min_col - 1])

                if i == 0:
                    con.spPr.ln.solidFill = ColorChoice(prstClr="red")
                elif i == 1:
                    con.spPr.ln.solidFill = ColorChoice(prstClr="green")
                elif i == 2:
                    con.spPr.ln.solidFill = ColorChoice(prstClr="blue")

                chart.series.append(con)

            ws.add_chart(chart, "N3")

        wb.save(filePath)

    @staticmethod
    def addHeaderOfColumnNames(filePath=None, colNameList=None, colUnitList=None):

        if colNameList != None and colUnitList == None:
            header = ""
            for i in range(len(colNameList)):
                if i != len(colNameList) - 1:
                    header += colNameList[i] + ","
                else:
                    header += colNameList[i]
            header += "\n"
        elif colNameList != None and colUnitList != None:
            header = ""
            for i in range(len(colNameList)):
                if i != len(colNameList) - 1:
                    header += colNameList[i] + "[" + colUnitList[i] + "],"
                else:
                    header += colNameList[i] + "[" + colUnitList[i] + "]"
            header += "\n"

        FileHandlerAscii.addLineOnTop(filePath=filePath, line=header)

    @staticmethod
    def getColumnTypeList(filePath=None, skiprows=None):
        tab = np.loadtxt(fname=filePath, skiprows=skiprows, delimiter=",")
        if type(tab) is np.ndarray:
            columnTypeList = []
            irow = 0
            for icol in range(tab.shape[1]):
                value = tab[irow][icol]
                columnTypeList.append(type(value))
            del tab
            return columnTypeList
        else:
            return None

    @staticmethod
    def writeFromDict(tableDict=None, filePath=None, index=False):
        df = pd.DataFrame.from_dict(tableDict)
        ext = FileHandlerAscii.getExtension(filePath=filePath)
        if ext == "csv":
            df.to_csv(filePath, index=index)
        elif ext == "xls" or ext=='xlsx':
            df.to_excel(filePath, index=index)

    @staticmethod
    def replace_none_with_empty(obj):
        if isinstance(obj, dict):
            return {k: TableHandler.replace_none_with_empty(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [TableHandler.replace_none_with_empty(item) for item in obj]
        elif obj is None:
            return ""
        else:
            return obj

    @staticmethod
    def readAsDict(filePath=None, keep_default_na=False):
        ext = FileHandlerAscii.getExtension(filePath=filePath)
        if ext == "csv":
            df       = pd.read_csv(filePath, keep_default_na=keep_default_na)
            tab_dict = df.to_dict()
            return TableHandler.replace_none_with_empty(obj=tab_dict)
        elif ext == "xls" or ext=='xlsx':
            df       = pd.read_excel(filePath,keep_default_na=keep_default_na)
            tab_dict = df.to_dict()
            return TableHandler.replace_none_with_empty(obj=tab_dict)

    @staticmethod
    def readAsNumpyTable(filePath=None, skiprows=None, hasHeader=False):
        return TableHandler.openAsNumpyTable(filePath=filePath, skiprows=skiprows, hasHeader=hasHeader)

    @staticmethod
    def openAsNumpyTable(filePath=None, skiprows=None, hasHeader=False):
        ext = FileHandlerAscii.getExtension(filePath=filePath)
        if ext == "csv":
            if hasHeader:
                colNameList = TableHandler.getHeaderAsColumnNameList(filePath=filePath)
                colTypeList = TableHandler.getColumnTypeList(filePath=filePath, skiprows=skiprows)
                if len(colNameList) != len(colTypeList):
                    print('Error: length of colNameList and colTypeList should be the same', file=sys.stderr)
                    return None
                tab = np.loadtxt(fname=filePath, skiprows=skiprows, delimiter=",",
                                dtype=[(colNameList[icol], colTypeList[icol]) for icol in range(len(colNameList))])
            return tab
        return None

    @staticmethod
    def readExcelDataAsPandasDfList(filePathExcel=None,withSheetNameList=False,sheetNameListOnly=False):

        ext = FilePathHandler.getExtension(filePath=filePathExcel)

        if ext == 'xls' or ext == 'xlsx':

            if sheetNameListOnly:
                return None, TableHandler.getExcelSheetNameList(filePathExcel=filePathExcel)

            input_file = pd.ExcelFile(filePathExcel)
            input_sheet_name = input_file.sheet_names

            df_list = []
            for sheet in input_sheet_name:
                df_list.append(input_file.parse(sheet))

            if withSheetNameList==False:
                return df_list
            else:
                return df_list, TableHandler.getExcelSheetNameList(filePathExcel=filePathExcel)

        elif ext == 'csv':
            
            assert(withSheetNameList==False)

            df_list = [pd.read_csv(filePathExcel)]

    @staticmethod
    def readExcelDataAsPandasDfSingle(filePathExcel=None):
        ext = FilePathHandler.getExtension(filePath=filePathExcel)
        if ext == 'xls' or ext == 'xlsx':
            return pd.read_excel(filePathExcel)
        elif ext == 'csv':
            return pd.read_csv(filePathExcel)
    
    @staticmethod
    def getExcelSheetNameList(filePathExcel=None):
        ext = FilePathHandler.getExtension(filePath=filePathExcel)
        if ext == "xls" or ext == "xlsx":
            return pd.ExcelFile(filePathExcel).sheet_names
        if ext == "csv":
            return ["Sheet1"]
        return []

    @staticmethod
    def writePandasDataframeListAsExcel(dfList=None, filePathExcel=None, sheetNameList=None):
        with pd.ExcelWriter(filePathExcel, engine='openpyxl', mode='w') as writer:
            for i in range(len(dfList)):
                df = dfList[i]
                if sheetNameList != None:
                    df.to_excel(writer, sheet_name=sheetNameList[i], index=False)#, encoding='utf-8')
                else:
                    df.to_excel(writer, sheet_name="Sheet" + str(i + 1), index=False)#, encoding='utf-8')

    @staticmethod
    def getPandasDfListFromCsvs(filePathListCsv=None):
        dfList = []
        sheetNameList = []
        for filePathCsv in filePathListCsv:
            sheetNameList.append(FileHandlerAscii.getFileNameBase(filePath=filePathCsv))
            tab = TableHandler.openAsNumpyTable(filePath=filePathCsv, skiprows=1, hasHeader=True)
            df = pd.DataFrame(tab)
            dfList.append(df)
        return dfList, sheetNameList
    
    @staticmethod
    def convertExcelFilesToExcel(filePathListExcel=None,filePathExcel=None,sheetNameList=None):
        dfList = []
        for filePath in filePathListExcel:
            dfList.append(TableHandler.readExcelDataAsPandasDfSingle(filePathExcel=filePath))
        TableHandler.writePandasDataframeListAsExcel(dfList=dfList,filePathExcel=filePathExcel,sheetNameList=sheetNameList)
    
    @staticmethod
    def convertCsvFilesToExcel(filePathListCsv=None, filePathExcel=None, sheetNameList=None):
        dfList, sheetNameList_Loc = TableHandler.getPandasDfListFromCsvs(filePathListCsv=filePathListCsv)
        if sheetNameList == None:
            sheetNameList = sheetNameList_Loc
        TableHandler.writePandasDataframeListAsExcel(dfList, filePathExcel=filePathExcel, sheetNameList=sheetNameList)

    @staticmethod
    def getHeaderAsColumnNameList(filePath=None):
        ext = FileHandlerAscii.getExtension(filePath=filePath)
        if ext == "csv":
            f = open(filePath)
            header = f.readline()
            headerList = header.split(',')

            for i in range(len(headerList)):
                headerItem = headerList[i]
                headerList[i] = headerItem.replace("\n", "")

            return headerList
        elif ext == "xlsx" or ext == "xls":
            tab = pd.read_excel(filePath, header=0)
            headerList = list(tab.columns)
            del tab
            return headerList

    @staticmethod
    def getUnitFromColumnName(columnName=None):
        columnName = re.split('\[', columnName)
        unitBuff = re.split('\]', columnName[1])
        unit = unitBuff[0]
        return unit

    @staticmethod
    def getNameFromColumnName(columnName=None):
        colName = re.split('\[', columnName)
        name = colName[0]
        return name

    @staticmethod
    def getUnitListFromColumnNameList(columnNameList):
        columnUnitList = []
        for i in range(len(columnNameList)):
            unit = TableHandler.getUnitFromColumnName(columnName=columnNameList[i])
            columnUnitList.append(unit)
        return columnUnitList

    @staticmethod
    def getNameListFromColumnNameList(columnNameList):
        varNameList = []
        for i in range(len(columnNameList)):
            name = TableHandler.getNameFromColumnName(columnName=columnNameList[i])
            varNameList.append(name)
        return varNameList

    @staticmethod
    def setData(tab=None, icol=None, irow=None, value=None):
        if type(tab) is np.ndarray:
            tab[irow][icol] = value
        elif type(tab) is pd.core.frame.DataFrame:
            tab.iat[irow, icol] = value

    @staticmethod
    def setDataRow(tab=None, irow=None, valueList=None):
        for icol in range(len(valueList)):
            TableHandler.setData(tab, icol=icol, irow=irow, value=valueList[icol])

    @staticmethod
    def convertCsvToExcel(filePathCsv=None, filePathExcel=None, index=None):
        df = pd.read_csv(filePathCsv)
        fileNameBase = FileHandlerAscii.getFileNameBase(filePath=filePathCsv)
        if filePathExcel == None:
            filePathExcel = fileNameBase + ".xlsx"
        df.to_excel(filePathExcel, index=index, sheet_name=fileNameBase)


    @staticmethod
    def getMaximumForAllColumnData(tab=None):
        if type(tab) is np.ndarray:
            valueMaxList = []
            for icol in range(tab.shape[1]):
                colData = tab[:, icol]
                valueMaxList.append(np.max(colData))
            return valueMaxList
        elif type(tab) is pd.core.frame.DataFrame:
            df = tab
            valueMaxList = []
            for icol in range(df.shape[1]):
                colData = df[df.columns[icol]]
                valueMaxList.append(np.max(colData))
            return valueMaxList

    @staticmethod
    def getMinimumForAllColumnData(tab=None):
        if type(tab) is np.ndarray:
            valueMinList = []
            for icol in range(tab.shape[1]):
                colData = tab[:, icol]
                valueMinList.append(np.min(colData))
            return valueMinList
        elif type(tab) is pd.core.frame.DataFrame:
            df = tab
            valueMinList = []
            for icol in range(df.shape[1]):
                colData = df[df.columns[icol]]
                valueMinList.append(np.min(colData))
            return valueMinList

    @staticmethod
    def getNormalizeColumnData(tab=None, icol=None, valueMin=None, valueMax=None):
        if type(tab) is np.ndarray:
            colData = tab[:, icol]
            colDataNorm = (colData - valueMin) / (valueMax - valueMin)
            return colDataNorm
        elif type(tab) is pd.core.frame.DataFrame:
            df = tab
            colData = df[df.columns[icol]]
            colDataNorm = (colData - valueMin) / (valueMax - valueMin)
            return colDataNorm

    @staticmethod
    def getNormalizedTable(tab=None, valueMinList=None, valueMaxList=None):

        if valueMinList is None:
            valueMinList = TableHandler.getMinimumForAllColumnData(tab=tab)

        if valueMaxList is None:
            valueMaxList = TableHandler.getMaximumForAllColumnData(tab=tab)

        if type(tab) is np.ndarray:
            tabRes = tab.copy()
            for icol in range(tabRes.shape[1]):
                colDataNorm = TableHandler.getNormalizeColumnData(tab=tabRes, icol=icol, valueMin=valueMinList[icol],
                                                                valueMax=valueMaxList[icol])
                for irow in range(tabRes.shape[0]):
                    tabRes[irow][icol] = colDataNorm[irow]
            return tabRes
        elif type(tab) is pd.core.frame.DataFrame:
            df = tab
            dfRes = None
            for icol in range(df.shape[1]):
                itemName = df.columns[icol]
                colDataNorm = TableHandler.getNormalizeColumnData(tab=df, icol=icol, valueMin=valueMinList[icol],
                                                                valueMax=valueMaxList[icol])
                if icol == 0:
                    dfRes = pd.DataFrame(colDataNorm, columns=[itemName])
                else:
                    dfRes = pd.concat([dfRes, pd.DataFrame(colDataNorm, columns=[itemName])], axis=1)
            return dfRes

    @staticmethod
    def getSubsetTable(tab=None, colNames=None, returnType=None, dtype=None):
        tab_sub = TableHandler.getColumnDataByNames(tab=tab, colNames=colNames)
        if type(tab) is np.ndarray:
            if returnType is np.ndarray:
                return tab_sub
            elif returnType is pd.core.frame.DataFrame:
                return pd.DataFrame(tab_sub)
        elif type(tab) is pd.core.frame.DataFrame:
            if returnType is np.ndarray:
                return tab_sub.to_numpy(dtype=dtype)
            elif returnType is pd.core.frame.DataFrame:
                return tab_sub
    
    @staticmethod
    def readTableFromBase64EncodedContent(content=None, fileType=None, returnType=None):
        import base64
        from io import BytesIO
        import pandas as pd
        decoded_content = base64.b64decode(content)

        if fileType == 'text/csv':
            df = pd.read_csv (BytesIO(decoded_content)) 
        elif fileType == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or fileType=='application/vnd.ms-excel':
            df = pd.read_excel(BytesIO(decoded_content)) 

        if returnType is np.ndarray:
            return df.to_numpy()
        elif returnType is pd.core.frame.DataFrame:
            return df

    @staticmethod
    def checkFileHasData(filePath, sheet_name=0):
        import csv
        ext = FilePathHandler.getExtension(filePath=filePath)
        if ext == 'csv':
            with open(filePath, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader, None)  # Read the header
                return any(row for row in reader)  # Check if there's any data row
        elif ext == 'xls' or ext == 'xlsx':
            df = pd.read_excel(filePath, sheet_name=sheet_name)
            return not df.empty  # Check if the DataFrame is not empty
        else:
            raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")

    @staticmethod
    def convertTableDictToListFormat(tableDict=None):
        headers  = list(tableDict.keys())
        num_rows = len(next(iter(tableDict.values())))
        rows     = [[tableDict[key][i] for key in headers] for i in range(num_rows)]
        result   = {
            "headers": headers,
            "rows": rows
        }
        return result
    
    @staticmethod
    def convertTableDictToMarkdownTable(tableDict=None):
        headers = list(tableDict.keys())
        rows    = zip(*[list(tableDict[key].values()) for key in headers])
        # Create the markdown table
        markdown_table = "| " + " | ".join(headers) + " |\n"
        markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            markdown_table += "| " + " | ".join(map(str, row)) + " |\n"
        return markdown_table

    # =========================================================================
    # Universal format conversion for LLM agents
    # =========================================================================
    # Lets agents pick the best representation for the situation:
    #   - Markdown : visual scan / small tables / chat embedding
    #   - CSV      : bulk data / pandas-friendly / efficient tokens
    #   - JSON     : nested / per-record lookup / multi-dim
    #   - Excel    : human deliverable
    # `recommend()` and `summarize()` help pick automatically.
    # =========================================================================

    @staticmethod
    def _toPandas(data):
        """Normalize any common tabular input into a pandas DataFrame.

        Accepted shapes:
          - pandas.DataFrame                         → returned as-is
          - numpy.ndarray (structured or 2D)         → DataFrame
          - dict[str, list|ndarray]                  → column-list dict
          - dict[str, dict[int, value]]              → pandas to_dict() style
          - dict (single record)                     → single-row DataFrame
          - list[dict]                               → records → DataFrame
          - list[list|tuple]                         → rows → DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            if not data:
                return pd.DataFrame()
            first_val = next(iter(data.values()))
            if isinstance(first_val, dict):
                # {col: {idx: val}} — pandas to_dict() form
                return pd.DataFrame(data)
            if isinstance(first_val, (list, tuple, np.ndarray, pd.Series)):
                return pd.DataFrame(data)
            # treat as single record
            return pd.DataFrame([data])
        if isinstance(data, list):
            if not data:
                return pd.DataFrame()
            if isinstance(data[0], dict):
                return pd.DataFrame(data)
            return pd.DataFrame(data)
        # last resort
        return pd.DataFrame(data)

    @staticmethod
    def toMarkdown(data, maxRows=None, floatFmt=".4g",
                   alignNumeric=True, includeIndex=False,
                   filePath=None):
        """Convert to a markdown table string (and optionally write to file).

        Parameters
        ----------
        maxRows : int | None
            If given and len(data) > maxRows, show first maxRows//2 + last
            maxRows//2 rows separated by `...` (truncated view).
        floatFmt : str
            Python format spec for floats (e.g. ".4g", ".3f").
        alignNumeric : bool
            Right-align numeric columns (`---:`).
        includeIndex : bool
            Whether to render the index column.
        filePath : str | None
            If given, write the result to this path.
        """
        df = TableHandler._toPandas(data)
        if includeIndex and df.index.name is None:
            df = df.reset_index().rename(columns={"index": "idx"})
        elif includeIndex:
            df = df.reset_index()

        # Build alignment row
        align = []
        for col in df.columns:
            if alignNumeric and pd.api.types.is_numeric_dtype(df[col]):
                align.append("---:")
            else:
                align.append("---")

        def fmt(v):
            if isinstance(v, float):
                return format(v, floatFmt)
            return str(v)

        # Optional truncation
        truncated = False
        if maxRows is not None and len(df) > maxRows:
            half = maxRows // 2
            head = df.head(half)
            tail = df.tail(maxRows - half)
            rows_data = list(head.itertuples(index=False, name=None)) + \
                        [tuple(["..."] * len(df.columns))] + \
                        list(tail.itertuples(index=False, name=None))
            truncated = True
        else:
            rows_data = list(df.itertuples(index=False, name=None))

        lines = []
        lines.append("| " + " | ".join(map(str, df.columns)) + " |")
        lines.append("| " + " | ".join(align) + " |")
        for row in rows_data:
            lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
        out = "\n".join(lines) + "\n"
        if truncated:
            out += f"\n*({len(df)} rows total, showing {maxRows})*\n"
        if filePath:
            Path = __import__("pathlib").Path
            p = Path(filePath); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(out, encoding="utf-8")
        return out

    @staticmethod
    def toCsv(data, filePath=None, includeIndex=False,
              floatFmt=None, sep=","):
        """Convert to CSV. Returns the string; optionally writes to file."""
        df = TableHandler._toPandas(data)
        kwargs = dict(index=includeIndex, sep=sep)
        if floatFmt is not None:
            kwargs["float_format"] = lambda v: format(v, floatFmt)
        if filePath:
            Path = __import__("pathlib").Path
            Path(filePath).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filePath, **kwargs)
            return filePath
        return df.to_csv(**kwargs)

    @staticmethod
    def toJson(data, filePath=None, orient="records", indent=None):
        """Convert to JSON. `orient` can be:
          "records"  : [{col1:v1, col2:v2}, ...]   — best for per-record lookup
          "columns"  : {col1:[...], col2:[...]}    — most compact
          "index"    : {idx: {col: val}}            — verbose
          "split"    : {columns:[...], data:[[...]]}
          "table"    : json-table-schema (verbose)
        """
        df = TableHandler._toPandas(data)
        if filePath:
            Path = __import__("pathlib").Path
            Path(filePath).parent.mkdir(parents=True, exist_ok=True)
            df.to_json(filePath, orient=orient, indent=indent)
            return filePath
        return df.to_json(orient=orient, indent=indent)

    @staticmethod
    def toExcelFile(data, filePath, sheetName="Sheet1",
                    includeIndex=False):
        """Convert to .xlsx file."""
        df = TableHandler._toPandas(data)
        Path = __import__("pathlib").Path
        Path(filePath).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(filePath, sheet_name=sheetName, index=includeIndex)
        return filePath

    @staticmethod
    def convert(data, toFormat="markdown", **kwargs):
        """Unified entry point: convert to the requested format.

        toFormat : "markdown" | "csv" | "json" | "excel" | "pandas" | "numpy"
        Extra kwargs are passed through to the underlying converter.
        """
        f = toFormat.lower()
        if f in ("md", "markdown"):
            return TableHandler.toMarkdown(data, **kwargs)
        if f == "csv":
            return TableHandler.toCsv(data, **kwargs)
        if f == "json":
            return TableHandler.toJson(data, **kwargs)
        if f in ("xlsx", "excel"):
            return TableHandler.toExcelFile(data, **kwargs)
        if f in ("pandas", "df"):
            return TableHandler._toPandas(data)
        if f in ("numpy", "np"):
            return TableHandler.getNumpyTable(TableHandler._toPandas(data))
        raise ValueError(f"Unknown toFormat: {toFormat}")

    @staticmethod
    def recommend(data, llmContext=True):
        """Recommend the most agent-friendly format for a given table.

        Heuristic:
          - small  (≤30 rows × ≤8 cols)        → markdown
          - medium (≤200 rows)                  → csv
          - large  (>200 rows or >10 cols)      → "summary" (markdown head/tail
                                                  + stats) plus csv file pointer

        Returns dict with keys: nRows, nCols, sizeClass, recommended,
        rationale.
        """
        df = TableHandler._toPandas(data)
        n, m = df.shape
        if n == 0:
            return dict(nRows=0, nCols=m, sizeClass="empty",
                        recommended="markdown",
                        rationale="empty table")
        if n <= 30 and m <= 8:
            cls = "small"; rec = "markdown"
            why = "small enough for visual scan; markdown best for chat embedding"
        elif n <= 200 and m <= 10:
            cls = "medium"; rec = "csv"
            why = "medium size; csv is compact, pandas-friendly"
        else:
            cls = "large"; rec = "summary"
            why = (f"{n}×{m} too large for inline scan; "
                   "use summary markdown + CSV file pointer")
        return dict(nRows=n, nCols=m, sizeClass=cls,
                    recommended=rec, rationale=why)

    @staticmethod
    def summarize(data, headRows=5, tailRows=5,
                  includeStats=True, floatFmt=".4g"):
        """Produce a Markdown-friendly summary for large tables.

        Layout:
          - shape line
          - head (`headRows` rows) as markdown
          - `...`
          - tail (`tailRows` rows)
          - describe() statistics for numeric columns
        """
        df = TableHandler._toPandas(data)
        n, m = df.shape
        parts = [f"**Table shape**: {n} rows × {m} cols"]

        if n <= headRows + tailRows + 2:
            parts.append(TableHandler.toMarkdown(df, floatFmt=floatFmt))
        else:
            head_md = TableHandler.toMarkdown(df.head(headRows),
                                                floatFmt=floatFmt).rstrip()
            tail_md = TableHandler.toMarkdown(df.tail(tailRows),
                                                floatFmt=floatFmt)
            tail_lines = tail_md.split("\n")[2:]  # drop header lines
            parts.append(head_md + "\n| " + " | ".join(["..."]*m) + " |\n" +
                          "\n".join(tail_lines))

        if includeStats:
            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                desc = num_df.describe().T[["count","mean","std","min","max"]]
                parts.append("\n**Numeric column stats**:\n")
                parts.append(TableHandler.toMarkdown(
                    desc.reset_index().rename(columns={"index": "column"}),
                    floatFmt=floatFmt))

        return "\n".join(parts)

if __name__ == '__main__':
    # _/_/_/_/ Initialize Numpy Array with Column Name
    # (1) Initialize Matrix
    tab = TableHandler.getInitializedNumpyTable(numColumn=3, numRow=10, colNames=['V1', 'V2', 'V3'])

    tab['V1'][0] = 1.0
    tab['V2'][0] = 2.0
    tab['V3'][0] = 3.0
    print(tab)

    # (2) Read from file
    tab[0][0] = 1.0
    tab[1][0] = 0.5
    tab[2][0] = 0.25
    tab[1][1] = 2.0
    tab[2][2] = 3.0

    # Access column data by Column Name (NumpyTable)  [Done]
    colData = tab['V1']
    print(colData)

    # Access column data by Column Index (NumpyTable) [Done]
    icolSt = 0
    icolEd = 1
    colData = TableHandler.getColumnDataByIndex(tab, icolSt, icolEd)
    print("#### Access column data by column index")
    print(colData)

    # Access sub-array (NumpyTable) [Done]
    icolSt = 0
    icolEd = 1
    irowSt = 0
    irowEd = 3
    subArray = TableHandler.getSubarray(tab, icolSt, icolEd, irowSt, irowEd)
    print("#### Access sub-array")
    print(subArray)

    # Display Table Contents [----]
    # Display Table Column Names [----]
    print(tab)
    print(tab.dtype.names)

    # _/_/_/_/ Convert to Pandas DataFrame [Done]
    import pandas as pd

    # Convet to pandas table
    df = pd.DataFrame(tab)
    # Display Pandas Dataframe
    print(df)

    # _/_/_/_/ Convert Pandas DataFrame to Numpy Table [Done]
    tab_np = TableHandler.getNumpyTable(df)
    print("\n >>Converted to Numpy Table:\n")
    print(tab_np)
    print(tab_np.dtype.names)

    # _/_/_/_/ Convert Numpy Table to Pandas DataFrame [Done]
    tab_pd = TableHandler.getPandasTable(tab)
    print("\n >>Converted to Pandas Dataframe:\n")
    print(tab_pd)
