namespace Italbytz.ML.Data;

public static class Data
{
    private static IDataset? _iris;
    private static IDataset? _breastCancerWisconsinDiagnostic;
    private static IDataset? _wineQuality;
    private static IDataset? _heartDisease;
    private static IDataset? _heartDiseaseBinary;
    private static IDataset? _adult;
    private static IDataset? _studentPerformance;
    private static IDataset? _automobile;
    private static IDataset? _balanceScale;
    private static IDataset? _carEvaluation;
    private static IDataset? _lenses;
    private static IDataset? _npha;
    private static IDataset? _solarflare1;
    private static IDataset? _obesityLevels;
    private static IDataset? _cdcDiabetes;

    public static IDataset CDCDiabetes =>
        _cdcDiabetes ??= new CDCDiabetesDataset();

    public static IDataset ObesityLevels =>
        _obesityLevels ??= new ObesityLevelsDataset();

    public static IDataset SolarFlare1 =>
        _solarflare1 ??= new SolarFlareDataset();

    public static IDataset NPHA =>
        _npha ??= new NPHADataset();

    public static IDataset Lenses =>
        _lenses ??= new LensesDataset();

    public static IDataset CarEvaluation =>
        _carEvaluation ??= new CarEvaluationDataset();

    public static IDataset BalanceScale =>
        _balanceScale ??= new BalanceScaleDataset();

    public static IDataset Automobile =>
        _automobile ??= new AutomobileDataset();

    public static IDataset StudentPerformance =>
        _studentPerformance ??= new StudentPerformanceDataset();

    public static IDataset Iris => _iris ??= new IrisDataset();

    public static IDataset BreastCancerWisconsinDiagnostic =>
        _breastCancerWisconsinDiagnostic ??=
            new BreastCancerWisconsinDiagnosticDataset();

    public static IDataset WineQuality =>
        _wineQuality ??= new WineQualityDataset();

    public static IDataset HeartDisease =>
        _heartDisease ??= new HeartDiseaseDataset();

    public static IDataset HeartDiseaseBinary =>
        _heartDiseaseBinary ??= new HeartDiseaseBinaryDataset();

    public static IDataset Adult =>
        _adult ??= new AdultIncomeDataset();
}