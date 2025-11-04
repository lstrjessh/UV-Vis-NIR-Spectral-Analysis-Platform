
Imports System.Security.Permissions
Imports System.Runtime.InteropServices
Imports System.Runtime.InteropServices.ComTypes

Public NotInheritable Class GraphTools

    Private Sub New()
        MyBase.New()
    End Sub

    ' ========= Add a filter to a DirectShow Graph using its CLSID
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Function AddFilterFromClsid(ByVal graphBuilder As IGraphBuilder, ByVal clsid As Guid, ByVal name As String) As IBaseFilter
        Dim hr As Integer = 0
        Dim filter As IBaseFilter = Nothing
        Try
            Dim type As Type = type.GetTypeFromCLSID(clsid)
            filter = CType(Activator.CreateInstance(type), IBaseFilter)
            graphBuilder.AddFilter(filter, name)
        Catch e As System.Exception
            Release_IbaseFilter(filter)
        End Try
        Return filter
    End Function

    ' ========= Add a filter to a DirectShow Graph using its name
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Function AddFilterByName(ByVal graphBuilder As IGraphBuilder, ByVal deviceCategory As Guid, ByVal friendlyName As String) As IBaseFilter
        If graphBuilder Is Nothing Then Return Nothing
        Dim devices() As DsDevice = DsDevice.GetDevicesOfCat(deviceCategory)
        Dim filter As IBaseFilter = Nothing
        Dim i As Integer = 0
        Dim name As String
        Do While (i < devices.Length)
            name = i.ToString & " " & devices(i).Name
            If name IsNot Nothing Then
                If name.Equals(friendlyName, StringComparison.InvariantCultureIgnoreCase) Then
                    ' using the return value DirectShowLib does not reports an error 
                    Dim hr As Int32 = TryCast(graphBuilder, IFilterGraph2).AddSourceFilterForMoniker(devices(i).Mon, Nothing, friendlyName, filter)
                    Exit Do
                End If
            End If
            i = (i + 1)
        Loop
        Return filter
    End Function

    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Function AddFilterByPartialName(ByVal graphBuilder As IGraphBuilder, ByVal deviceCategory As Guid, ByVal PartialName As String) As IBaseFilter
        If graphBuilder Is Nothing Then Return Nothing
        Dim devices() As DsDevice = DsDevice.GetDevicesOfCat(deviceCategory)
        Dim filter As IBaseFilter = Nothing
        Dim i As Integer = 0
        Dim name As String
        PartialName = LCase(PartialName)
        Do While (i < devices.Length)
            name = i.ToString & " " & devices(i).Name
            If name IsNot Nothing Then
                If InStr(LCase(name), PartialName) > 0 Then
                    ' using the return value DirectShowLib does not reports an error
                    Dim hr As Int32 = CType(graphBuilder, IFilterGraph2).AddSourceFilterForMoniker(devices(i).Mon, Nothing, name, filter)
                    Exit Do
                End If
            End If
            i = (i + 1)
        Loop
        Return filter
    End Function

    ' ========= Add a filter to a DirectShow Graph using its Moniker's device path
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Function AddFilterByDevicePath(ByVal graphBuilder As IGraphBuilder, ByVal devicePath As String, ByVal name As String) As IBaseFilter
        Dim hr As Integer = 0
        Dim filter As IBaseFilter = Nothing

        Dim bindCtx As IBindCtx = Nothing
        Dim moniker As IMoniker = Nothing

        Dim eaten As Integer
        If (graphBuilder Is Nothing) Then
            Throw New ArgumentNullException("graphBuilder")
        End If
        Try
            hr = NativeMethods.CreateBindCtx(0, bindCtx)
            Marshal.ThrowExceptionForHR(hr)
            hr = NativeMethods.MkParseDisplayName(bindCtx, devicePath, eaten, moniker)
            Marshal.ThrowExceptionForHR(hr)
            hr = CType(graphBuilder, IFilterGraph2).AddSourceFilterForMoniker(moniker, bindCtx, name, filter)
            DsError.ThrowExceptionForHR(hr)
        Catch e As System.Exception
            ' An error occur. Just returning null...
        Finally
            ReleaseComObject(bindCtx) : bindCtx = Nothing
            ReleaseComObject(moniker) : moniker = Nothing
        End Try
        Return filter
    End Function

    ' ========= Find a filter in a DirectShow Graph using its name
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Function FindFilterByName(ByVal graphBuilder As IGraphBuilder, ByVal filterName As String) As IBaseFilter
        Dim hr As Integer = 0
        Dim filter As IBaseFilter = Nothing
        Dim enumFilters As IEnumFilters = Nothing
        If (graphBuilder Is Nothing) Then
            Throw New ArgumentNullException("graphBuilder")
        End If
        hr = graphBuilder.EnumFilters(enumFilters)
        If (hr = 0) Then
            Dim filters(0) As IBaseFilter

            While (enumFilters.Next(filters.Length, filters, IntPtr.Zero) = 0)
                Dim filterInfo As FilterInfo = Nothing
                hr = filters(0).QueryFilterInfo(filterInfo)
                If (hr = 0) Then
                    ReleaseComObject(filterInfo.pGraph) : filterInfo.pGraph = Nothing
                    If filterInfo.achName.Equals(filterName) Then
                        filter = filters(0)
                        Exit While
                    End If
                End If
                Release_IbaseFilter(filters(0))

            End While
            ReleaseComObject(enumFilters) : enumFilters = Nothing
        End If
        Return filter
    End Function

    ' ========= Find a filter in a DirectShow Graph using its CLSID
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Function FindFilterByClsid(ByVal graphBuilder As IGraphBuilder, ByVal filterClsid As Guid) As IBaseFilter
        Dim hr As Integer = 0
        Dim filter As IBaseFilter = Nothing
        Dim enumFilters As IEnumFilters = Nothing
        If (graphBuilder Is Nothing) Then
            Throw New ArgumentNullException("graphBuilder")
        End If
        hr = graphBuilder.EnumFilters(enumFilters)
        If (hr = 0) Then
            Dim filters(0) As IBaseFilter

            While (enumFilters.Next(filters.Length, filters, IntPtr.Zero) = 0)
                Dim clsid As Guid
                hr = filters(0).GetClassID(clsid)
                If ((hr = 0) AndAlso (clsid = filterClsid)) Then
                    filter = filters(0)
                    Exit While
                End If
                Release_IbaseFilter(filters(0))

            End While
            ReleaseComObject(enumFilters) : enumFilters = Nothing
        End If
        Return filter
    End Function

    ' ========= Render a filter's pin in a DirectShow Graph
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Function RenderPin(ByVal graphBuilder As IGraphBuilder, ByVal source As IBaseFilter, ByVal pinName As String) As Boolean
        Dim hr As Integer = 0
        If (graphBuilder Is Nothing) Then
            Throw New ArgumentNullException("graphBuilder")
        End If
        If (source Is Nothing) Then
            Throw New ArgumentNullException("source")
        End If
        Dim pin As IPin = DsFindPin.ByName(source, pinName)
        If (Not (pin) Is Nothing) Then
            hr = graphBuilder.Render(pin)
            Release_IPin(pin)
            Return (hr >= 0)
        End If
        Return False
    End Function

    ' ========= Disconnect all pins on a given filter
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Sub DisconnectPins(ByVal filter As IBaseFilter)
        Dim hr As Integer = 0
        If (filter Is Nothing) Then
            Throw New ArgumentNullException("filter")
        End If
        Dim enumPins As IEnumPins = Nothing
        Dim pins(0) As IPin

        hr = filter.EnumPins(enumPins)
        DsError.ThrowExceptionForHR(hr)
        Try

            While (enumPins.Next(pins.Length, pins, IntPtr.Zero) = 0)
                Try
                    hr = pins(0).Disconnect
                    DsError.ThrowExceptionForHR(hr)
                Finally
                    Release_IPin(pins(0))
                End Try

            End While
        Finally
            ReleaseComObject(enumPins) : enumPins = Nothing
        End Try
    End Sub

    ' ========= Disconnect pins of all the filters in a DirectShow Graph
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Sub DisconnectAllPins(ByVal graphBuilder As IGraphBuilder)
        Dim hr As Integer = 0
        Dim enumFilters As IEnumFilters = Nothing
        If (graphBuilder Is Nothing) Then
            Throw New ArgumentNullException("graphBuilder")
        End If
        hr = graphBuilder.EnumFilters(enumFilters)
        DsError.ThrowExceptionForHR(hr)
        Try
            Dim filters(0) As IBaseFilter

            While (enumFilters.Next(filters.Length, filters, IntPtr.Zero) = 0)
                Try
                    DisconnectPins(filters(0))
                Catch e As System.Exception
                End Try
                Release_IbaseFilter(filters(0))
            End While
        Finally
            ReleaseComObject(enumFilters) : enumFilters = Nothing
        End Try
    End Sub

    ' ========= Remove and release all filters from a DirectShow Graph
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Sub RemoveAllFilters(ByVal graphBuilder As IGraphBuilder)
        Dim hr As Integer = 0
        Dim enumFilters As IEnumFilters = Nothing
        Dim filtersArray As ArrayList = New ArrayList
        If (graphBuilder Is Nothing) Then
            Throw New ArgumentNullException("graphBuilder")
        End If
        hr = graphBuilder.EnumFilters(enumFilters)
        DsError.ThrowExceptionForHR(hr)
        Try
            Dim filters(0) As IBaseFilter

            While (enumFilters.Next(filters.Length, filters, IntPtr.Zero) = 0)
                filtersArray.Add(filters(0))

            End While
        Finally
            ReleaseComObject(enumFilters) : enumFilters = Nothing
        End Try
        For Each filter As IBaseFilter In filtersArray
            hr = graphBuilder.RemoveFilter(filter)
            Release_IbaseFilter(filter)
        Next
    End Sub

    ' ========= Save a DirectShow Graph to a GRF file
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Sub SaveGraphFile(ByVal graphBuilder As IGraphBuilder, ByVal fileName As String)
        Dim hr As Integer = 0
        Dim storage As IStorage = Nothing

        Dim stream As IStream = Nothing

        If (graphBuilder Is Nothing) Then
            Throw New ArgumentNullException("graphBuilder")
        End If
        Try
            hr = NativeMethods.StgCreateDocfile(fileName, (STGM.Create _
                            Or (STGM.Transacted _
                            Or (STGM.ReadWrite Or STGM.ShareExclusive))), 0, storage)
            Marshal.ThrowExceptionForHR(hr)

            hr = storage.CreateStream("ActiveMovieGraph", _
                                      STGM.Write Or STGM.Create Or STGM.ShareExclusive _
                                      , 0 _
                                      , 0 _
                                      , stream)

            Marshal.ThrowExceptionForHR(hr)
            hr = CType(graphBuilder, IPersistStream).Save(stream, True)
            Marshal.ThrowExceptionForHR(hr)
            hr = storage.Commit(STGC.DefaultValue)
            Marshal.ThrowExceptionForHR(hr)
        Finally
            ReleaseComObject(stream) : stream = Nothing
            ReleaseComObject(storage) : storage = Nothing
        End Try
    End Sub

    ' ========= Load a DirectShow Graph from a file
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Sub LoadGraphFile(ByVal graphBuilder As IGraphBuilder, ByVal fileName As String)
        Dim hr As Integer = 0
        Dim storage As IStorage = Nothing

        Dim stream As IStream = Nothing

        If (graphBuilder Is Nothing) Then
            Throw New ArgumentNullException("graphBuilder")
        End If
        Try
            If (NativeMethods.StgIsStorageFile(fileName) <> 0) Then
                Throw New ArgumentException
            End If
            hr = NativeMethods.StgOpenStorage(fileName, Nothing, (STGM.Transacted _
                            Or (STGM.Read Or STGM.ShareDenyWrite)), IntPtr.Zero, 0, storage)
            Marshal.ThrowExceptionForHR(hr)
            hr = storage.OpenStream("ActiveMovieGraph", IntPtr.Zero, (STGM.Read Or STGM.ShareExclusive), 0, stream)
            Marshal.ThrowExceptionForHR(hr)
            hr = CType(graphBuilder, IPersistStream).Load(stream)
            Marshal.ThrowExceptionForHR(hr)
        Finally
            ReleaseComObject(stream) : stream = Nothing
            ReleaseComObject(storage) : storage = Nothing
        End Try
    End Sub

    ' ========= Check if a DirectShow filter can display Property Pages
    Public Shared Function HasPropertyPages(ByVal filter As IBaseFilter) As Boolean
        If (filter Is Nothing) Then
            Throw New ArgumentNullException("filter")
        End If
        Return (Not (TryCast(filter, ISpecifyPropertyPages)) Is Nothing)
    End Function

    ' ========= Display Property Pages of a given DirectShow filter
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Sub ShowFilterPropertyPage(ByVal filter As IBaseFilter, ByVal parent As IntPtr)
        Dim hr As Integer = 0
        Dim filterInfo As FilterInfo = Nothing
        Dim caGuid As DsCAUUID
        Dim objs() As Object
        If (filter Is Nothing) Then
            Throw New ArgumentNullException("filter")
        End If
        If HasPropertyPages(filter) Then
            hr = filter.QueryFilterInfo(filterInfo)
            DsError.ThrowExceptionForHR(hr)
            ReleaseComObject(filterInfo.pGraph) : filterInfo.pGraph = Nothing
            hr = CType(filter, ISpecifyPropertyPages).GetPages(caGuid)
            DsError.ThrowExceptionForHR(hr)
            Try
                objs = New Object((1) - 1) {}
                objs(0) = filter
                NativeMethods.OleCreatePropertyFrame(parent, 0, 0, filterInfo.achName, objs.Length, objs, caGuid.cElems, caGuid.pElems, 0, 0, IntPtr.Zero)
            Finally
                Marshal.FreeCoTaskMem(caGuid.pElems)
            End Try
        End If
    End Sub

    ' ========= Check if a COM Object is available
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Function IsThisComObjectInstalled(ByVal clsid As Guid) As Boolean
        Dim retval As Boolean = False
        Try
            Dim t As Type = Type.GetTypeFromCLSID(clsid)
            Dim o As Object = Activator.CreateInstance(t)
            retval = True
            ReleaseComObject(o) : o = Nothing
        Catch e As System.Exception
        End Try
        Return retval
    End Function

    ' ========= Check if the Video Mixing Renderer 9 Filter is available
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Function IsVMR9Present() As Boolean
        Return IsThisComObjectInstalled(GetType(VideoMixingRenderer9).GUID)
    End Function

    ' ========= Check if the Video Mixing Renderer 7 Filter is available
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Shared Function IsVMR7Present() As Boolean
        Return IsThisComObjectInstalled(GetType(VideoMixingRenderer).GUID)
    End Function

    ' ========= Connect pins from two filters
    Public Overloads Shared Sub ConnectFilters(ByVal graphBuilder As IGraphBuilder, ByVal upFilter As IBaseFilter, ByVal sourcePinName As String, ByVal downFilter As IBaseFilter, ByVal destPinName As String, ByVal useIntelligentConnect As Boolean)
        If (graphBuilder Is Nothing) Then
            Throw New ArgumentNullException("graphBuilder")
        End If
        If (upFilter Is Nothing) Then
            Throw New ArgumentNullException("upFilter")
        End If
        If (downFilter Is Nothing) Then
            Throw New ArgumentNullException("downFilter")
        End If
        Dim destPin As IPin
        Dim sourcePin As IPin
        sourcePin = DsFindPin.ByName(upFilter, sourcePinName)
        If (sourcePin Is Nothing) Then
            Throw New ArgumentException(("The source filter has no pin called : " + sourcePinName), sourcePinName)
        End If
        destPin = DsFindPin.ByName(downFilter, destPinName)
        If (destPin Is Nothing) Then
            Throw New ArgumentException(("The downstream filter has no pin called : " + destPinName), destPinName)
        End If
        Try
            ConnectFilters(graphBuilder, sourcePin, destPin, useIntelligentConnect)
        Finally
            Release_IPin(sourcePin)
            Release_IPin(destPin)
        End Try
    End Sub

    ' ========= Connect pins from two filters
    <SecurityPermission(SecurityAction.LinkDemand, UnmanagedCode:=True)> _
    Public Overloads Shared Sub ConnectFilters(ByVal graphBuilder As IGraphBuilder, ByVal sourcePin As IPin, ByVal destPin As IPin, ByVal useIntelligentConnect As Boolean)
        Dim hr As Integer = 0
        If (graphBuilder Is Nothing) Then
            Throw New ArgumentNullException("graphBuilder")
        End If
        If (sourcePin Is Nothing) Then
            Throw New ArgumentNullException("sourcePin")
        End If
        If (destPin Is Nothing) Then
            Throw New ArgumentNullException("destPin")
        End If
        If useIntelligentConnect Then
            hr = graphBuilder.Connect(sourcePin, destPin)
            DsError.ThrowExceptionForHR(hr)
        Else
            hr = graphBuilder.ConnectDirect(sourcePin, destPin, Nothing)
            DsError.ThrowExceptionForHR(hr)
        End If
    End Sub
End Class

<Flags()> _
Enum STGM
    Read = 0
    Write = 1
    ReadWrite = 2
    ShareDenyNone = 64
    ShareDenyRead = 48
    ShareDenyWrite = 32
    ShareExclusive = 16
    Priority = 262144
    Create = 4096
    Convert = 131072
    FailIfThere = 0
    Direct = 0
    Transacted = 65536
    NoScratch = 1048576
    NoSnapShot = 2097152
    Simple = 134217728
    DirectSWMR = 4194304
    DeleteOnRelease = 67108864
End Enum

<Flags()> _
Enum STGC
    DefaultValue = 0
    Overwrite = 1
    OnlyIfCurrent = 2
    DangerouslyCommitMerelyToDiskCache = 4
    Consolidate = 8
End Enum

<Guid("0000000b-0000-0000-C000-000000000046"), _
 InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Interface IStorage

    <PreserveSig()> _
    Function CreateStream(ByVal pwcsName As String, ByVal grfMode As STGM, ByVal reserved1 As Integer, ByVal reserved2 As Integer, ByRef ppstm As IStream) As Integer

    <PreserveSig()> _
    Function OpenStream(ByVal pwcsName As String, ByVal reserved1 As IntPtr, ByVal grfMode As STGM, ByVal reserved2 As Integer, ByRef ppstm As IStream) As Integer

    <PreserveSig()> _
    Function CreateStorage(ByVal pwcsName As String, ByVal grfMode As STGM, ByVal reserved1 As Integer, ByVal reserved2 As Integer, ByRef ppstg As IStorage) As Integer

    <PreserveSig()> _
    Function OpenStorage(ByVal pwcsName As String, ByVal pstgPriority As IStorage, ByVal grfMode As STGM, ByVal snbExclude As Integer, ByVal reserved As Integer, ByRef ppstg As IStorage) As Integer

    <PreserveSig()> _
    Function CopyTo(ByVal ciidExclude As Integer, ByVal rgiidExclude() As Guid, ByVal snbExclude() As String, ByVal pstgDest As IStorage) As Integer

    <PreserveSig()> _
    Function MoveElementTo(ByVal pwcsName As String, ByVal pstgDest As IStorage, ByVal pwcsNewName As String, ByVal grfFlags As STGM) As Integer

    <PreserveSig()> _
    Function Commit(ByVal grfCommitFlags As STGC) As Integer

    <PreserveSig()> _
    Function Revert() As Integer

    <PreserveSig()> _
    Function EnumElements(ByVal reserved1 As Integer, ByVal reserved2 As IntPtr, ByVal reserved3 As Integer, ByRef ppenum As Object) As Integer

    <PreserveSig()> _
    Function DestroyElement(ByVal pwcsName As String) As Integer

    <PreserveSig()> _
    Function RenameElement(ByVal pwcsOldName As String, ByVal pwcsNewName As String) As Integer

    <PreserveSig()> _
    Function SetElementTimes(ByVal pwcsName As String, _
                             ByVal filetime As System.Runtime.InteropServices.ComTypes.FILETIME, _
                             ByVal patime As System.Runtime.InteropServices.ComTypes.FILETIME, _
                             ByVal pmtime As System.Runtime.InteropServices.ComTypes.FILETIME) As Integer

    <PreserveSig()> _
    Function SetClass(ByVal clsid As Guid) As Integer

    <PreserveSig()> _
    Function SetStateBits(ByVal grfStateBits As Integer, ByVal grfMask As Integer) As Integer

    <PreserveSig()> _
    Function Stat(ByRef pStatStg As System.Runtime.InteropServices.ComTypes.STATSTG, ByVal grfStatFlag As Int32) As Integer
End Interface


NotInheritable Class NativeMethods

    Private Sub New()
        MyBase.New()
    End Sub

    Public Declare Function CreateBindCtx Lib "ole32.dll" (ByVal reserved As Integer, ByRef ppbc As IBindCtx) As Integer

    Public Declare Function MkParseDisplayName Lib "ole32.dll" (ByVal pcb As IBindCtx, ByVal szUserName As String, ByRef pchEaten As Integer, ByRef ppmk As IMoniker) As Integer

    Public Declare Function OleCreatePropertyFrame Lib "oleaut32.dll" (ByVal hwndOwner As IntPtr, ByVal x As Integer, ByVal y As Integer, ByVal lpszCaption As String, ByVal cObjects As Integer, ByVal ppUnk() As Object, ByVal cPages As Integer, ByVal pPageClsID As IntPtr, ByVal lcid As Integer, ByVal dwReserved As Integer, ByVal pvReserved As IntPtr) As Integer

    Public Declare Function StgCreateDocfile Lib "ole32.dll" (ByVal pwcsName As String, ByVal grfMode As STGM, ByVal reserved As Integer, ByRef ppstgOpen As IStorage) As Integer

    Public Declare Function StgIsStorageFile Lib "ole32.dll" (ByVal pwcsName As String) As Integer

    Public Declare Function StgOpenStorage Lib "ole32.dll" (ByVal pwcsName As String, ByVal pstgPriority As IStorage, ByVal grfMode As STGM, ByVal snbExclude As IntPtr, ByVal reserved As Integer, ByRef ppstgOpen As IStorage) As Integer
End Class




Module GraphHelperFunctions


    '' -----------------------------------------------------------------------------------------------
    ''   TEST a GRAPH for ALL CONNECTED FILTERS   ( time 2 to 10 mS ) 
    '' -----------------------------------------------------------------------------------------------
    'Friend Function IsGraphAllConnected(ByVal gBuilder As IFilterGraph2) As Boolean
    '    Dim EnumFlt As IEnumFilters = Nothing
    '    Dim EnumPns As IEnumPins = Nothing
    '    Dim filter(0) As IBaseFilter
    '    Dim pin(0) As IPin
    '    Dim pinNext As IPin = Nothing
    '    Dim ConnectedPins As Int32 = 0
    '    Dim UnconnectedFilters As Int32 = 0
    '    If gBuilder Is Nothing Then Return False
    '    ' ----------------------------------------------------------
    '    If gBuilder.EnumFilters(EnumFlt) = 0 Then
    '        ' ------------------------------------------------------
    '        While EnumFlt.Next(1, filter, 0) = 0
    '            ' --------------------------------------------------
    '            If filter(0).EnumPins(EnumPns) = 0 Then
    '                ConnectedPins = 0
    '                ' ----------------------------------------------
    '                While EnumPns.Next(1, pin, 0) = 0
    '                    ' ------------------------------------------
    '                    If pin(0).ConnectedTo(pinNext) = 0 Then
    '                        If pinNext IsNot Nothing Then
    '                            ConnectedPins += 1
    '                            Release_IPin(pinNext)
    '                            ' ----------------------------------
    '                        End If
    '                    End If
    '                    Release_IPin(pin(0))
    '                    ' ------------------------------------------
    '                End While
    '                ReleaseComObject(EnumPns) : EnumPns = Nothing
    '                ' ----------------------------------------------
    '                If ConnectedPins = 0 Then UnconnectedFilters += 1
    '            End If
    '            Release_IbaseFilter(filter(0))
    '            ' --------------------------------------------------
    '        End While
    '        ReleaseComObject(EnumFlt) : EnumFlt = Nothing
    '        ' ------------------------------------------------------
    '    End If
    '    ' ----------------------------------------------------------
    '    If UnconnectedFilters > 0 Then
    '        Return False
    '    Else
    '        Return True
    '    End If
    'End Function


    'Friend Sub Render_FilterOutputPin(ByVal GraphBuilder As IGraphBuilder, ByVal filter As IBaseFilter, ByVal pin As Int32)
    '    Dim pinOut As IPin
    '    pinOut = DsFindPin.ByDirection(filter, PinDirection.Output, pin)
    '    GraphBuilder.Render(pinOut)
    '    Release_IPin(pinOut)
    'End Sub

    'Friend Sub Render_FilterPinByName(ByVal GraphBuilder As IGraphBuilder, ByVal filter As IBaseFilter, ByVal pinName As String)
    '    Dim pinOut As IPin
    '    pinOut = DsFindPin.ByName(filter, pinName)
    '    GraphBuilder.Render(pinOut)
    '    Release_IPin(pinOut)
    'End Sub


    'Friend Sub ConnectBDABoardFilters(ByVal graphBuilder As IFilterGraph2, _
    '                                  ByVal CaptureGraphBuilder As ICaptureGraphBuilder2, _
    '                                      ByVal tunerfilter As IBaseFilter, _
    '                                      ByVal mpeg2demuxfilter As IBaseFilter, _
    '                                      ByRef demodulatorfilter As IBaseFilter, _
    '                                      ByRef capturefilter As IBaseFilter)

    '    Dim hr As Int32 = 0
    '    Dim devices As DsDevice()

    '    Try
    '        ' ------------------------------------------ trying to connect this filter to the MPEG-2 Demux 
    '        hr = CaptureGraphBuilder.RenderStream(Nothing, Nothing, tunerfilter, Nothing, mpeg2demuxfilter)
    '        If hr >= 0 Then
    '            ' -------------------------------------- this is a one filter model 
    '            demodulatorfilter = Nothing
    '            capturefilter = Nothing
    '            Return
    '        Else
    '            ' -------------------------------------- enumerate BDA Receiver Components category 
    '            ' -------------------------------------- to found a filter connecting to the tuner and the MPEG2 Demux 
    '            devices = DsDevice.GetDevicesOfCat(FilterCategory.BDAReceiverComponentsCategory)
    '            For i As Integer = 0 To devices.Length - 1

    '                Dim tmp As IBaseFilter = Nothing

    '                hr = GraphBuilder.AddSourceFilterForMoniker(devices(i).Mon, Nothing, devices(i).Name, tmp)
    '                'DsError.ThrowExceptionForHR(hr)

    '                hr = CaptureGraphBuilder.RenderStream(Nothing, Nothing, tunerfilter, Nothing, tmp)
    '                If hr = 0 Then
    '                    ' ------------------------------- found 
    '                    capturefilter = tmp
    '                    '-------------------------------- connect it to the MPEG-2 Demux 
    '                    hr = CaptureGraphBuilder.RenderStream(Nothing, Nothing, capturefilter, Nothing, mpeg2demuxfilter)
    '                    If hr >= 0 Then
    '                        ' This second filter connect both with the tuner and the demux. 
    '                        ' This is a capture filter... 
    '                        Return
    '                    Else
    '                        ' This second filter connect with the tuner but not with the demux. 
    '                        ' This is in fact a demodulator filter. We now must find the true capture filter... 

    '                        demodulatorfilter = capturefilter
    '                        capturefilter = Nothing

    '                        ' saving the Demodulator's DevicePath to avoid creating it twice. 
    '                        Dim demodulatorDevicePath As String = devices(i).DevicePath

    '                        Dim j As Integer = 0
    '                        While i < devices.Length
    '                            If devices(j).DevicePath.Equals(demodulatorDevicePath) Then
    '                                Continue While
    '                            End If

    '                            hr = GraphBuilder.AddSourceFilterForMoniker(devices(i).Mon, Nothing, devices(i).Name, tmp)
    '                            'DsError.ThrowExceptionForHR(hr)

    '                            hr = CaptureGraphBuilder.RenderStream(Nothing, Nothing, demodulatorfilter, Nothing, tmp)
    '                            If hr = 0 Then
    '                                ' ------------------------------------- found 
    '                                capturefilter = tmp

    '                                ' Connect it to the MPEG-2 Demux 
    '                                hr = CaptureGraphBuilder.RenderStream(Nothing, Nothing, capturefilter, Nothing, mpeg2demuxfilter)
    '                                If hr >= 0 Then
    '                                    ' This second filter connect both with the demodulator and the demux. 
    '                                    ' This is a true capture filter... 
    '                                    Return
    '                                End If
    '                            Else
    '                                ' Try another... 
    '                                hr = GraphBuilder.RemoveFilter(tmp)
    '                                ReleaseComObject(tmp) : tmp = Nothing
    '                            End If
    '                            j += 1
    '                        End While
    '                        ' 
    '                        ' We have a tuner and a capture/demodulator that don't connect with the demux 
    '                        ' and we found no additionals filters to build a working filters chain. 
    '                        'Form_MsgBox.OK("Can't find a valid BDA filter chain")
    '                    End If
    '                Else
    '                    ' Try another... 
    '                    hr = GraphBuilder.RemoveFilter(tmp)
    '                    ReleaseComObject(tmp) : tmp = Nothing
    '                End If
    '            Next
    '            ' 
    '            ' We have a tuner that connect to the Network Provider BUT not with the demux 
    '            ' and we found no additionals filters to build a working filters chain. 
    '            'Form_MsgBox.OK("Can't find a valid BDA filter chain")
    '        End If
    '    Finally
    '    End Try
    'End Sub


    ' ====================================================================================================
    '  RELEASE COM OBJECT    ( DO A MANUAL "Obj = Nothing" AFTER THIS CALL )
    ' ====================================================================================================
    '
    '   with the option "STRICT" it is not possible to use ByRef with generic "Objects"
    '   so: it is not possible to set "Obj = Nothing"
    '   so: "Nothing" MUST be done manually AFTER calling this
    '
    ' ====================================================================================================
    Friend Sub ReleaseComObject(ByVal obj As Object)
        If obj IsNot Nothing Then
            Marshal.ReleaseComObject(obj)
        End If
    End Sub


    ' ====================================================================================================
    '  RELEASE COM OBJECT - SPECIALIZED CALLS    ( AUTOMATIC "Obj = Nothing" )
    ' ====================================================================================================
    Friend Sub Release_IbaseFilter(ByRef obj As IBaseFilter)
        If obj IsNot Nothing Then
            Marshal.ReleaseComObject(obj)
            obj = Nothing
        End If
    End Sub
    Friend Sub Release_IPin(ByRef obj As IPin)
        If obj IsNot Nothing Then
            Marshal.ReleaseComObject(obj)
            obj = Nothing
        End If
    End Sub

End Module
