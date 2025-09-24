using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;

namespace Italbytz.ML.Data;

public interface IDataset
{
    public string? LabelColumnName { get; }
    public IDataView DataView { get; }

    public IColumnProperties[] ColumnProperties { get; }

    public string FilePrefix { get; }

    public bool AllowQuoting { get; }
    public bool AllowSparse { get; }
    public char Separator { get; }
    public char DecimalMarker { get; }
    public bool HasHeader { get; }
    public bool TrimWhitespace { get; }
    public bool ReadMultilines { get; }
    public char EscapeChar { get; }
    public bool MissingRealsAsNaNs { get; }

    public IEnumerable<TrainValidateTestFileNames> GetTrainValidateTestFiles(
        string saveFolderPath,
        string? samplingKeyColumnName = null,
        double validateFraction = 0.15,
        double testFraction = 0.15,
        int[]? seeds = null
    );

    public IEstimator<ITransformer> BuildPipeline(MLContext mlContext,
        IEstimator<ITransformer> estimator,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard);

    public IEstimator<ITransformer> BuildPreprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard);

    public IEstimator<ITransformer>? BuildPostprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard);

    protected IDataView LoadFromTextFile<TModelInput>(
        string path,
        char? separatorChar = null,
        bool? hasHeader = null,
        bool? allowQuoting = null,
        bool? trimWhitespace = null,
        bool? allowSparse = null);

    public IDataView LoadFromTextFile(
        string path,
        char? separatorChar = null,
        bool? hasHeader = null,
        bool? allowQuoting = null,
        bool? trimWhitespace = null,
        bool? allowSparse = null);
}