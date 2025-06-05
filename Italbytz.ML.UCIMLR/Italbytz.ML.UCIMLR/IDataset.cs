using Italbytz.ML.Data;
using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;

namespace Italbytz.ML.Data;

public interface IDataset
{
    public string? LabelColumnName { get; }
    public IDataView DataView { get; }

    public IColumnProperties[] ColumnProperties { get; }

    public string FilePrefix { get; }

    public IEnumerable<TrainValidateTestFileNames> GetTrainValidateTestFiles(
        string saveFolderPath,
        string? samplingKeyColumnName = null,
        double validateFraction = 0.15,
        double testFraction = 0.15,
        int[]? seeds = null
    );

    public IEstimator<ITransformer> BuildPipeline(MLContext mlContext,
        ScenarioType scenarioType, IEstimator<ITransformer> estimator);

    protected IDataView LoadFromTextFile<TModelInput>(
        string path,
        char separatorChar = TextLoaderDefaults.Separator,
        bool hasHeader = TextLoaderDefaults.HasHeader,
        bool allowQuoting = TextLoaderDefaults.AllowQuoting,
        bool trimWhitespace = TextLoaderDefaults.TrimWhitespace,
        bool allowSparse = TextLoaderDefaults.AllowSparse);

    public IDataView LoadFromTextFile(
        string path,
        char separatorChar = TextLoaderDefaults.Separator,
        bool hasHeader = TextLoaderDefaults.HasHeader,
        bool allowQuoting = TextLoaderDefaults.AllowQuoting,
        bool trimWhitespace = TextLoaderDefaults.TrimWhitespace,
        bool allowSparse = TextLoaderDefaults.AllowSparse);

    internal static class TextLoaderDefaults
    {
        internal const bool AllowQuoting = false;
        internal const bool AllowSparse = false;
        internal const char Separator = '\t';
        internal const char DecimalMarker = '.';
        internal const bool HasHeader = false;
        internal const bool TrimWhitespace = false;
        internal const bool ReadMultilines = false;
        internal const char EscapeChar = '"';
        internal const bool MissingRealsAsNaNs = false;
    }
}