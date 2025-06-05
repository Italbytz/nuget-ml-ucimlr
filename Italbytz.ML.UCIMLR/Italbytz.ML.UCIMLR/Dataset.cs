using System.Text.Json;
using System.Text.Json.Serialization;
using Italbytz.ML.Data;
using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;

namespace Italbytz.ML.Data;

public abstract class Dataset<TModelInput> : IDataset
{
    private IColumnProperties[]? _columnProperties;

    private IDataView? _dataView;

    protected virtual string ColumnPropertiesString { get; }

    protected virtual string ResourceName { get; }
    public virtual string FilePrefix { get; }

    public IColumnProperties[] ColumnProperties =>
        _columnProperties ??= GetColumnProperties();

    public virtual string? LabelColumnName { get; }
    public IDataView DataView => _dataView ??= LoadDataView();

    public IEnumerable<TrainValidateTestFileNames> GetTrainValidateTestFiles(
        string saveFolderPath,
        string? samplingKeyColumnName = null, double validateFraction = 0.15,
        double testFraction = 0.15, int[]? seeds = null)
    {
        return DataView.GenerateTrainValidateTestCsvs(
            saveFolderPath, FilePrefix, samplingKeyColumnName,
            validateFraction, testFraction, seeds);
    }

    public abstract IEstimator<ITransformer> BuildPipeline(MLContext mlContext,
        ScenarioType scenarioType, IEstimator<ITransformer> estimator);

    public abstract IDataView LoadFromTextFile(string path,
        char separatorChar = IDataset.TextLoaderDefaults.Separator,
        bool hasHeader = IDataset.TextLoaderDefaults.HasHeader,
        bool allowQuoting = IDataset.TextLoaderDefaults.AllowQuoting,
        bool trimWhitespace = IDataset.TextLoaderDefaults.TrimWhitespace,
        bool allowSparse = IDataset.TextLoaderDefaults.AllowSparse);


    public IDataView LoadFromTextFile<TModelInput>(string path,
        char separatorChar = IDataset.TextLoaderDefaults.Separator,
        bool hasHeader = IDataset.TextLoaderDefaults.HasHeader,
        bool allowQuoting = IDataset.TextLoaderDefaults.AllowQuoting,
        bool trimWhitespace = IDataset.TextLoaderDefaults.TrimWhitespace,
        bool allowSparse = IDataset.TextLoaderDefaults.AllowSparse)
    {
        var mlContext = new MLContext();
        // Load the dataset from the specified path
        var data = mlContext.Data.LoadFromTextFile<TModelInput>(
            path, separatorChar, hasHeader, allowQuoting, trimWhitespace,
            allowSparse);
        return data;
    }

    private IDataView? LoadDataView()
    {
        var stream = GetStream();
        var tempFile = Path.GetTempFileName();
        using var fileStream = File.Create(tempFile);
        stream?.CopyTo(fileStream);
        fileStream.Flush();
        fileStream.Close();
        var data = LoadFromTextFile<TModelInput>(tempFile, ',', true);
        return data ??
               throw new InvalidOperationException("Failed to load data");
    }

    private Stream GetStream()
    {
        var assembly = typeof(Dataset<TModelInput>).Assembly;
        var stream = assembly.GetManifestResourceStream(ResourceName);
        return stream;
    }

    private ColumnPropertiesV5[] GetColumnProperties()
    {
        var options = new JsonSerializerOptions
        {
            Converters =
            {
                new JsonStringEnumConverter()
            }
        };

        return
            JsonSerializer.Deserialize<ColumnPropertiesV5[]>(
                ColumnPropertiesString, options);
    }
}