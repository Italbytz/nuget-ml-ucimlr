using System.Text.Json;
using System.Text.Json.Serialization;
using Italbytz.ML.Data;
using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.UCIMLR;

public abstract class Dataset : IDataset
{
    private IColumnProperties[]? _columnProperties;

    private IDataView? _dataView;

    protected virtual string ColumnPropertiesString { get; }
    protected virtual string FilePrefix { get; }

    protected virtual string ResourceName { get; }

    public IColumnProperties[] ColumnProperties =>
        _columnProperties ??= GetColumnProperties();

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

    private IDataView? LoadDataView()
    {
        var stream = GetStream();
        var tempFile = Path.GetTempFileName();
        using var fileStream = File.Create(tempFile);
        stream?.CopyTo(fileStream);
        fileStream.Flush();
        fileStream.Close();
        var data = LoadFromTextFile(tempFile);
        return data ??
               throw new InvalidOperationException("Failed to load data");
    }

    protected abstract IDataView? LoadFromTextFile(string tempFile);

    private Stream GetStream()
    {
        var assembly = typeof(StaticData).Assembly;
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


    /// <summary>
    ///     Represents the input data schema for the Iris dataset used in ML.NET
    ///     models.
    /// </summary>
    protected class IrisModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"sepal length")]
        public float Sepal_length { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"sepal width")]
        public float Sepal_width { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"petal length")]
        public float Petal_length { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"petal width")]
        public float Petal_width { get; set; }

        [LoadColumn(4)] [ColumnName(@"class")] public string Class { get; set; }
    }
}