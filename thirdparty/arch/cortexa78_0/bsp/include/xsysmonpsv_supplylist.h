#ifndef XSYSMONPSV_SUPPLYLIST
#define XSYSMONPSV_SUPPLYLIST

/*
* The supply configuration table for sysmon
*/
typedef enum {
    VCCAUX,
    VCCINT,
    VCC_RAM,
    VCC_SOC,
    VCCAUX_LPD,
    VCC_PMC,
    VCC_PSFP,
    VCC_PSLP,
    VP_VN,
    VCCINT_MMI_MMI,
    EndList,
} XSysMonPsv_Supply;

#endif