
Imports System.Runtime.InteropServices
Imports System.Runtime.InteropServices.ComTypes

Namespace BDA

    <ComImport(), Guid("14EB8748-1753-4393-95AE-4F7E7A87AAD6")> _
    Public Class TIFLoad
    End Class


#If ALLOW_UNTESTED_INTERFACES Then

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("DFEF4A68-EE61-415f-9CCB-CD95F2F98A3A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_TIF_REGISTRATION
        <PreserveSig()> _
        Function RegisterTIFEx(<[In]()> ByVal pTIFInputPin As IPin, <Out()> ByRef ppvRegistrationContext As Integer, <Out(), MarshalAs(UnmanagedType.[Interface])> ByRef ppMpeg2DataControl As Object) As Integer

        <PreserveSig()> _
        Function UnregisterTIF(<[In]()> ByVal pvRegistrationContext As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("F9BAC2F9-4149-4916-B2EF-FAA202326862"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IMPEG2_TIF_CONTROL
        <PreserveSig()> _
        Function RegisterTIF(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pUnkTIF As Object, <[In](), Out()> ByRef ppvRegistrationContext As Integer) As Integer

        <PreserveSig()> _
        Function UnregisterTIF(<[In]()> ByVal pvRegistrationContext As Integer) As Integer

        <PreserveSig()> _
        Function AddPIDs(<[In]()> ByVal ulcPIDs As Integer, <[In]()> ByRef pulPIDs As Integer) As Integer

        <PreserveSig()> _
        Function DeletePIDs(<[In]()> ByVal ulcPIDs As Integer, <[In]()> ByRef pulPIDs As Integer) As Integer

        <PreserveSig()> _
        Function GetPIDCount(<Out()> ByRef pulcPIDs As Integer) As Integer

        <PreserveSig()> _
        Function GetPIDs(<Out()> ByRef pulcPIDs As Integer, <Out()> ByRef pulPIDs As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("A3B152DF-7A90-4218-AC54-9830BEE8C0B6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ITuneRequestInfo
        <PreserveSig()> _
        Function GetLocatorData(<[In]()> ByVal Request As ITuneRequest) As Integer

        <PreserveSig()> _
        Function GetComponentData(<[In]()> ByVal CurrentRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Function CreateComponentList(<[In]()> ByVal CurrentRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Function GetNextProgram(<[In]()> ByVal CurrentRequest As ITuneRequest, <Out()> ByRef TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Function GetPreviousProgram(<[In]()> ByVal CurrentRequest As ITuneRequest, <Out()> ByRef TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Function GetNextLocator(<[In]()> ByVal CurrentRequest As ITuneRequest, <Out()> ByRef TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Function GetPreviousLocator(<[In]()> ByVal CurrentRequest As ITuneRequest, <Out()> ByRef TuneRequest As ITuneRequest) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("EFDA0C80-F395-42c3-9B3C-56B37DEC7BB7"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IGuideDataEvent
        <PreserveSig()> _
        Function GuideDataAcquired() As Integer

        <PreserveSig()> _
        Function ProgramChanged(<[In]()> ByVal varProgramDescriptionID As Object) As Integer

        <PreserveSig()> _
        Function ServiceChanged(<[In]()> ByVal varProgramDescriptionID As Object) As Integer

        <PreserveSig()> _
        Function ScheduleEntryChanged(<[In]()> ByVal varProgramDescriptionID As Object) As Integer

        <PreserveSig()> _
        Function ProgramDeleted(<[In]()> ByVal varProgramDescriptionID As Object) As Integer

        <PreserveSig()> _
        Function ServiceDeleted(<[In]()> ByVal varProgramDescriptionID As Object) As Integer

        <PreserveSig()> _
        Function ScheduleDeleted(<[In]()> ByVal varProgramDescriptionID As Object) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("88EC5E58-BB73-41d6-99CE-66C524B8B591"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IGuideDataProperty
        <PreserveSig()> _
        Function get_Name(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

        <PreserveSig()> _
        Function get_Language(<Out()> ByRef idLang As Integer) As Integer

        <PreserveSig()> _
        Function get_Value(<Out()> ByRef pvar As Object) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("AE44423B-4571-475c-AD2C-F40A771D80EF"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IEnumGuideDataProperties
        <PreserveSig()> _
        Function [Next](<[In]()> ByVal celt As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=0)> ByVal ppprop As IGuideDataProperty(), <[In]()> ByVal pcelt As IntPtr) As Integer

        <PreserveSig()> _
        Function Skip(<[In]()> ByVal celt As Integer) As Integer

        <PreserveSig()> _
        Function Reset() As Integer

        <PreserveSig()> _
        Function Clone(<Out()> ByRef ppenum As IEnumGuideDataProperties) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("1993299C-CED6-4788-87A3-420067DCE0C7"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IEnumTuneRequests
        <PreserveSig()> _
        Function [Next](<[In]()> ByVal celt As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=0)> ByVal ppprop As ITuneRequest(), <[In]()> ByVal pcelt As IntPtr) As Integer

        <PreserveSig()> _
        Function Skip(<[In]()> ByVal celt As Integer) As Integer

        <PreserveSig()> _
        Function Reset() As Integer

        <PreserveSig()> _
        Function Clone(<Out()> ByRef ppenum As IEnumTuneRequests) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("61571138-5B01-43cd-AEAF-60B784A0BF93"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IGuideData
        <PreserveSig()> _
        Function GetServices(<Out()> ByRef ppEnumTuneRequests As IEnumTuneRequests) As Integer

        <PreserveSig()> _
        Function GetServiceProperties(<[In]()> ByVal pTuneRequest As ITuneRequest, <Out()> ByRef ppEnumProperties As IEnumGuideDataProperties) As Integer

        <PreserveSig()> _
        Function GetGuideProgramIDs(<Out()> ByRef pEnumPrograms As IEnumVARIANT) As Integer

        <PreserveSig()> _
        Function GetProgramProperties(<[In]()> ByVal varProgramDescriptionID As Object, <Out()> ByRef ppEnumProperties As IEnumGuideDataProperties) As Integer

        <PreserveSig()> _
        Function GetScheduleEntryIDs(<Out()> ByRef pEnumScheduleEntries As IEnumVARIANT) As Integer

        <PreserveSig()> _
        Function GetScheduleEntryProperties(<[In]()> ByVal varScheduleEntryDescriptionID As Object, <Out()> ByRef ppEnumProperties As IEnumGuideDataProperties) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("4764ff7c-fa95-4525-af4d-d32236db9e38"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IGuideDataLoader
        <PreserveSig()> _
        Function Init(<[In]()> ByVal pGuideStore As IGuideData) As Integer

        <PreserveSig()> _
        Function Terminate() As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("EE957C52-B0D0-4e78-8DD1-B87A08BFD893"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ITuneRequestInfoEx
        Inherits ITuneRequestInfo

        <PreserveSig()> _
        Shadows Function GetLocatorData(<[In]()> ByVal Request As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function GetComponentData(<[In]()> ByVal CurrentRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function CreateComponentList(<[In]()> ByVal CurrentRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function GetNextProgram(<[In]()> ByVal CurrentRequest As ITuneRequest, <Out()> ByRef TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function GetPreviousProgram(<[In]()> ByVal CurrentRequest As ITuneRequest, <Out()> ByRef TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function GetNextLocator(<[In]()> ByVal CurrentRequest As ITuneRequest, <Out()> ByRef TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function GetPreviousLocator(<[In]()> ByVal CurrentRequest As ITuneRequest, <Out()> ByRef TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Function CreateComponentListEx(ByVal CurrentRequest As ITuneRequest, <MarshalAs(UnmanagedType.IUnknown)> ByRef ppCurPMT As Object) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("7E47913A-5A89-423d-9A2B-E15168858934"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ISIInbandEPGEvent
        <PreserveSig()> _
        Function SIObjectEvent(ByVal pIDVB_EIT As IDVB_EIT2, ByVal dwTable_ID As Integer, ByVal dwService_ID As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("F90AD9D0-B854-4b68-9CC1-B2CC96119D85"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ISIInbandEPG
        <PreserveSig()> _
        Function StartSIEPGScan() As Integer

        <PreserveSig()> _
        Function StopSIEPGScan() As Integer

        <PreserveSig()> _
        Function IsSIEPGScanRunning(<MarshalAs(UnmanagedType.Bool)> ByRef bRunning As Boolean) As Integer
    End Interface

#End If

End Namespace
