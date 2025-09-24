using System.Text.Json;
using System.Text.Json.Serialization;
using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;

namespace Italbytz.ML.Data;

public abstract class Dataset<TModelInput> : IDataset
{
    private IColumnProperties[]? _columnProperties;

    private IDataView? _dataView;

    protected virtual string ColumnPropertiesString { get; }

    protected virtual string ResourceName { get; }

    public bool? HasHeader { get; set; } = true;

    public char? SeperatorChar { get; set; } = ',';
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


    public abstract IDataView LoadFromTextFile(string path,
        char? separatorChar = null,
        bool? hasHeader = null, bool? allowQuoting = null,
        bool? trimWhitespace = null, bool? allowSparse = null);

    public IDataView LoadFromTextFile<TModelInput1>(string path,
        char? separatorChar = null, bool? hasHeader = null,
        bool? allowQuoting = null, bool? trimWhitespace = null,
        bool? allowSparse = null)
    {
        separatorChar ??= IDataset.TextLoaderDefaults.Separator;
        hasHeader ??= IDataset.TextLoaderDefaults.HasHeader;
        allowQuoting ??= IDataset.TextLoaderDefaults.AllowQuoting;
        trimWhitespace ??= IDataset.TextLoaderDefaults.TrimWhitespace;
        allowSparse ??= IDataset.TextLoaderDefaults.AllowSparse;

        var mlContext = new MLContext();
        // Load the dataset from the specified path
        var data = mlContext.Data.LoadFromTextFile<TModelInput>(
            path, (char)separatorChar, (bool)hasHeader, (bool)allowQuoting,
            (bool)trimWhitespace,
            (bool)allowSparse);
        return data;
    }

    public IEstimator<ITransformer> BuildPipeline(MLContext mlContext,
        IEstimator<ITransformer> estimator,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        var pipeline =
            BuildPreprocessingPipeline(mlContext, scenarioType, processingType);
        pipeline = pipeline.Append(estimator);
        var postProcessing =
            BuildPostprocessingPipeline(mlContext, scenarioType,
                processingType);
        if (postProcessing !=
            null)
            pipeline = pipeline.Append(postProcessing);
        return pipeline;
    }

    public virtual IEstimator<ITransformer> BuildPreprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        var pipeline =
            AdditionalPreprocessingPipeline(mlContext, scenarioType,
                processingType);
        var featurization =
            BuildFeaturizationPipeline(mlContext, scenarioType, processingType);
        if (featurization != null)
            pipeline = pipeline.Append(featurization);
        var labelMapping =
            BuildLabelMappingPipeline(mlContext, scenarioType, processingType);
        if (labelMapping != null)
            pipeline = pipeline.Append(labelMapping);
        return pipeline;
    }

    public virtual IEstimator<ITransformer>? BuildPostprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        var additionalPostprocessing =
            AdditionalPostprocessingPipeline(mlContext, scenarioType,
                processingType);
        var labelRemapping =
            BuildLabelRemappingPipeline(mlContext, scenarioType,
                processingType);
        if (labelRemapping != null && additionalPostprocessing != null)
            return labelRemapping.Append(additionalPostprocessing);
        if (labelRemapping != null) return labelRemapping;
        return additionalPostprocessing ?? null;
    }


    protected abstract IEstimator<ITransformer> AdditionalPreprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard);

    protected virtual IEstimator<ITransformer>?
        AdditionalPostprocessingPipeline(
            MLContext mlContext,
            ScenarioType scenarioType = ScenarioType.Classification,
            ProcessingType processingType = ProcessingType.Standard)
    {
        return null;
    }

    protected virtual IEstimator<ITransformer>? BuildFeaturizationPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        return null;
    }

    protected virtual IEstimator<ITransformer>? BuildLabelMappingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        return null;
    }

    protected virtual IEstimator<ITransformer>? BuildLabelRemappingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        return null;
    }

    private IDataView? LoadDataView()
    {
        var stream = GetStream();
        var tempFile = Path.GetTempFileName();
        using var fileStream = File.Create(tempFile);
        stream?.CopyTo(fileStream);
        fileStream.Flush();
        fileStream.Close();
        var data =
            LoadFromTextFile<TModelInput>(tempFile, SeperatorChar, HasHeader);
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

public enum ProcessingType
{
    Standard,
    FeatureBinningAndCustomLabelMapping
}