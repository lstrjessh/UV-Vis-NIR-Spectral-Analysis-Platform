
Imports System.Runtime.InteropServices


Module Utils_DirectShow_EnumFilters


    ' ==============================================================================================================
    '   ENUM DIRECT-SHOW FILTERS BY MATCH
    ' ==============================================================================================================

    'Friend MEDIATYPE_Video As Guid = New Guid("{73646976-0000-0010-8000-00AA00389B71}")
    'Friend MEDIASUBTYPE_YV12 As Guid = New Guid("{32315659-0000-0010-8000-00AA00389B71}")
    'Friend MEDIASUBTYPE_YUY2 As Guid = New Guid("{32595559-0000-0010-8000-00AA00389B71}")
    'Friend MEDIASUBTYPE_NV12 As Guid = New Guid("{3231564E-0000-0010-8000-00AA00389B71}")
    'Friend MEDIASUBTYPE_MPEG2_VIDEO As Guid = New Guid("{E06D8026-DB46-11CF-B4D1-00805F6CBBEA}")
    'Friend MEDIASUBTYPE_RGB32 As Guid = New Guid("{E436EB7E-524F-11CE-9F53-0020AF0BA770}")
    'Friend MEDIASUBTYPE_MJPG As Guid = New Guid("{47504A4D-0000-0010-8000-00AA00389B71}")
    '
    'Friend MEDIATYPE_Audio As Guid = New Guid("{73647561-0000-0010-8000-00AA00389B71}")
    'Friend MEDIASUBTYPE_MPEG1_Audio As Guid = New Guid("{E436EB87-524F-11CE-9F53-0020AF0BA770}")
    'Friend MEDIASUBTYPE_MPEG2_Audio As Guid = New Guid("{E06D802B-DB46-11CF-B4D1-00805F6CBBEA}")

    'Friend GUID_NULL As Guid = New Guid("{00000000-0000-0000-0000-000000000000}")
    'Friend MEDIASUBTYPE_None As Guid = New Guid("{E436EB8E-524F-11CE-9F53-0020AF0BA770}")

    'Friend PIN_CATEGORY_ANALOGVIDEOIN As Guid = New Guid("{FB6C4283-0353-11D1-905F-0000C0CC16BA}")
    Friend PIN_CATEGORY_CAPTURE As Guid = New Guid("{FB6C4281-0353-11D1-905F-0000C0CC16BA}")
    'Friend PIN_CATEGORY_PREVIEW As Guid = New Guid("{FB6C4282-0353-11D1-905F-0000C0CC16BA}")
    'Friend PIN_CATEGORY_VIDEOPORT As Guid = New Guid("{FB6C4285-0353-11D1-905F-0000C0CC16BA}")

    
    Private CLSID_FilterMapper2 As Guid = New Guid("{CDA42200-BD88-11D0-BD4E-00A0C911CE86}")

    Friend Function EnumFiltersByMatch(ByVal ExactMatch As Boolean, _
                                          ByVal Merit As Merit, _
                                          ByVal InputNeeded As Boolean, _
                                          ByVal InputType As Guid, _
                                          ByVal InputSubType As Guid, _
                                          ByVal PinCategoryIn As Guid, _
                                          ByVal Render As Boolean, _
                                          ByVal OutputNeeded As Boolean, _
                                          ByVal OutputType As Guid, _
                                          ByVal OutputSubType As Guid, _
                                          ByVal PinCategoryOut As Guid) _
                                          As String()
        '
        Dim sarr(-1) As String
        EnumFiltersByMatch = sarr
        '
        Dim hr As Integer
        Dim comObj As Object = Nothing
        Dim enumDev As IFilterMapper2 = Nothing
        Dim enumMon As ComTypes.IEnumMoniker = Nothing
        Dim mon(0) As ComTypes.IMoniker
        '
        Dim InputTypes(1) As Guid
        InputTypes(0) = InputType
        InputTypes(1) = InputSubType
        '
        Dim OutputTypes(1) As Guid
        OutputTypes(0) = OutputType
        OutputTypes(1) = OutputSubType
        '
        Try
            Dim srvType As Type = Type.GetTypeFromCLSID(CLSID_FilterMapper2)
            comObj = Activator.CreateInstance(srvType)
            enumDev = CType(comObj, IFilterMapper2)
            ' ---------------------------------------------------- Enum matching filters
            hr = enumDev.EnumMatchingFilters(enumMon, 0, ExactMatch, Merit, _
                                            InputNeeded, 1, InputTypes, Nothing, PinCategoryIn, _
                                            Render, _
                                            OutputNeeded, 1, OutputTypes, Nothing, PinCategoryOut)

            Dim f As IntPtr
            Do
                ' ------------------------------------------------ Next filter
                hr = enumMon.Next(1, mon, f)
                If hr <> 0 OrElse mon(0) Is Nothing Then
                    Exit Do
                End If
                ' ------------------------------------------------ Add the filter
                ReDim Preserve sarr(UBound(sarr) + 1)
                sarr(UBound(sarr)) = GetFriendlyName(mon(0))
                ' ------------------------------------------------ Release mon(0)
                ReleaseComObject(mon(0)) : mon(0) = Nothing
            Loop
            Array.Sort(sarr)
            EnumFiltersByMatch = sarr
            '
        Catch ex As Exception
        Finally
            enumDev = Nothing
            ReleaseComObject(mon(0)) : mon(0) = Nothing
            ReleaseComObject(enumMon) : enumMon = Nothing
            ReleaseComObject(comObj) : comObj = Nothing
        End Try
        ''
    End Function



    ' ==============================================================================================================
    '   ENUM DIRECT-SHOW FILTERS By CATEGORY
    ' ==============================================================================================================

    'FilterCategory.VideoInputDevice
    'FilterCategory.AudioInputDevice
    'FilterCategory.VideoCompressorCategory
    'FilterCategory.AudioCompressorCategory
    'FilterCategory.AMKSTVTuner
    'FilterCategory.AMKSCrossbar

    Private CLSID_SystemDeviceEnum As Guid = New Guid("{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}")

    Friend Function EnumFiltersByCategory(ByVal category As Guid) As String()
        '
        Dim sarr(-1) As String
        EnumFiltersByCategory = sarr
        '
        Dim hr As Integer
        Dim comObj As Object = Nothing
        Dim enumDev As ICreateDevEnum = Nothing
        Dim enumMon As ComTypes.IEnumMoniker = Nothing
        Dim mon(0) As ComTypes.IMoniker
        '
        Try
            ' ---------------------------------------------------- Get the system device enumerator
            Dim srvType As Type = Type.GetTypeFromCLSID(CLSID_SystemDeviceEnum)
            comObj = Activator.CreateInstance(srvType)
            enumDev = CType(comObj, ICreateDevEnum)
            ' ---------------------------------------------------- Enum filters in category
            hr = enumDev.CreateClassEnumerator(category, enumMon, 0)
            Dim f As IntPtr
            Do
                ' ------------------------------------------------ Next filter
                hr = enumMon.Next(1, mon, f)
                If hr <> 0 OrElse mon(0) Is Nothing Then
                    Exit Do
                End If
                ' ------------------------------------------------ Add the filter
                ReDim Preserve sarr(UBound(sarr) + 1)
                sarr(UBound(sarr)) = (sarr.Length - 1).ToString & " " & GetFriendlyName(mon(0))
                ' ------------------------------------------------ Release mon(0)
                ReleaseComObject(mon(0)) : mon(0) = Nothing
            Loop
            Array.Sort(sarr)
            EnumFiltersByCategory = sarr
            '
        Catch ex As Exception
        Finally
            enumDev = Nothing
            ReleaseComObject(mon(0)) : mon(0) = Nothing
            ReleaseComObject(enumMon) : enumMon = Nothing
            ReleaseComObject(comObj) : comObj = Nothing
        End Try
        '
    End Function



    ' ---------------------------------------- Retrieve the human-readable name of the filter
    Private Function GetFriendlyName(ByVal mon As ComTypes.IMoniker) As String
        Dim bagObj As Object = Nothing
        Dim bag As IPropertyBag = Nothing
        Try
            Dim bagId As Guid = GetType(IPropertyBag).GUID
            mon.BindToStorage(Nothing, Nothing, bagId, bagObj)
            bag = CType(bagObj, IPropertyBag)
            Dim val As Object = ""
            bag.Read("FriendlyName", val, Nothing)
            Dim ret As String = CType(val, String)
            Return ret
        Catch
            Return ""
        Finally
            bag = Nothing
            ReleaseComObject(bagObj) : bagObj = Nothing
        End Try
    End Function


End Module
