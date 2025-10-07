# Test Case Generator

An AI-powered universal test case generator that creates unit tests for any programming language using various AI models. Works seamlessly on both local Jupyter environments and Google Colab.

## Features

- **Multi-Provider Support**: Generate tests using GPT, Claude, Gemini, or open-source HuggingFace models
- **Language Agnostic**: Works with any programming language
- **Flexible Configuration**: Customize number of tests and output token limits
- **User-Friendly UI**: Interactive Gradio interface for easy test generation
- **Environment Adaptive**: Automatically detects and configures for local or Colab environments
- **GPU Optimization**: Supports GPU acceleration with 4-bit quantization for HuggingFace models

## Supported Models

### GPT (OpenAI)
- `gpt-4o-mini`
- `gpt-4o`
- `gpt-3.5-turbo`

### Claude (Anthropic)
- `claude-3-5-haiku-latest`
- `claude-3-5-sonnet-latest`

### Gemini (Google)
- `gemini-2.0-flash-exp`
- `gemini-1.5-pro`
- `gemini-1.5-flash`

### HuggingFace (Open Source)
- `meta-llama/Llama-3.1-3B-Instruct`
- `microsoft/bitnet-b1.58-2B-4T`
- `tiiuae/Falcon-E-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

## Installation

### Local Setup

1. **Clone or download the repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the project root with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   GOOGLE_API_KEY=your_google_key_here
   HUGGINGFACE_API_KEY=your_huggingface_key_here
   GROQ_API_KEY=your_groq_key_here
   ```

   Note: You only need the API keys for the models you plan to use.

### Google Colab Setup

1. **Upload the notebook** to Google Colab

2. **Add your API keys** to Colab secrets:
   - Click on the key icon (🔑) in the left sidebar
   - Add secrets with names matching the environment variables above

3. **Run the notebook** - it will automatically install dependencies

## Usage

### Running Locally

1. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook test_case_generator_universal.ipynb
   ```

2. **Run all cells** in order

3. **Access the Gradio interface** at the local URL (typically `http://127.0.0.1:7865`)

### Running in Google Colab

1. **Upload the notebook** to Colab

2. **Run all cells** - the interface will launch with a public shareable link

### Using the Interface

1. **Select Model Category**: Choose from GPT, Claude, Gemini, or HuggingFace
2. **Select Specific Model**: Pick the exact model version you want to use
3. **Configure Options**:
   - Number of tests: How many test cases to generate (1-20)
   - Max output tokens: Control the length of the response (128-4096)
4. **Enter Your Code**: Paste the code you want to test or describe the requirements
5. **Generate**: Click the "Generate Tests" button
6. **Review Output**: The generated tests will appear in the code output panel

## Generated Test Format

The generator creates:
- ✅ Original code included at the top
- ✅ Comprehensive unit tests with comments
- ✅ Runnable code (no markdown formatting)
- ✅ Test runner/main function included
- ✅ Coverage for:
  - Happy path scenarios
  - Negative test cases
  - Boundary conditions
  - Edge cases

## Requirements

- Python 3.8+
- Internet connection (for API calls)
- Optional: CUDA-compatible GPU (for HuggingFace models)

## Dependencies

Core libraries:
- `gradio` - Interactive UI
- `openai` - GPT models
- `anthropic` - Claude models
- `google-generativeai` - Gemini models
- `transformers` - HuggingFace models
- `python-dotenv` - Environment configuration

See `requirements.txt` for complete list with versions.

## GPU Support

When using HuggingFace models, the system will automatically:
- Detect available GPU
- Enable 4-bit quantization for memory efficiency
- Use CPU fallback if no GPU is available

## Error Handling

The application includes comprehensive error handling:
- Missing API keys are detected and reported
- Invalid model selections are caught
- Generation errors are displayed in the output panel

## Tips for Best Results

1. **Provide Clear Code**: Well-formatted, complete code snippets work best
2. **Add Context**: Include brief comments or descriptions of complex logic
3. **Adjust Test Count**: Start with 5 tests, increase for complex functions
4. **Token Limits**: Increase max tokens for larger codebases or more comprehensive tests
5. **Model Selection**: 
   - Use GPT-4o or Claude for complex code
   - Use smaller models (GPT-3.5, Gemini Flash) for simple functions
   - Try HuggingFace models for privacy or offline testing

## Troubleshooting

**Issue**: "API key not configured"
- **Solution**: Verify your `.env` file contains the correct API key for the selected model

**Issue**: Out of memory error with HuggingFace models
- **Solution**: Try a smaller model or increase your system RAM/VRAM

**Issue**: Slow generation
- **Solution**: Use smaller models or reduce the max output tokens

**Issue**: Gradio interface not loading
- **Solution**: Check that all dependencies are installed correctly with `pip install -r requirements.txt`

## License

This project is provided as-is for educational and development purposes.

## Contributing

Contributions are welcome! Feel free to:
- Add support for new model providers
- Improve the system prompt for better test generation
- Enhance the UI with additional features
- Fix bugs or improve documentation

## Acknowledgments

Built with:
- [Gradio](https://gradio.app/) for the UI
- [OpenAI](https://openai.com/), [Anthropic](https://anthropic.com/), [Google AI](https://ai.google.dev/), and [HuggingFace](https://huggingface.co/) for AI models
- [PyTorch](https://pytorch.org/) and [Transformers](https://huggingface.co/docs/transformers) for model inference

