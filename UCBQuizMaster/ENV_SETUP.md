# Setting Up Your Environment Variables

## Creating the .env File

The application requires an OpenAI API key to function. You need to create a `.env` file with your API key.

### Option 1: Command Line (Recommended)

Run this command in the project directory:

```bash
echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
```

Replace `your_actual_api_key_here` with your actual OpenAI API key.

### Option 2: Manual Creation

1. Create a new file named `.env` (note the dot at the beginning)
2. Add this line to the file:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```
3. Save the file

## Getting Your OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Log in or create an account
3. Click "Create new secret key"
4. Copy the key (you won't be able to see it again!)
5. Paste it in your `.env` file

## Example .env File

Your `.env` file should look like this:

```
OPENAI_API_KEY=sk-proj-aBc123XyZ...
```

**Important**: 
- Don't add quotes around the key
- Don't add spaces before or after the `=`
- Keep this file secret - never commit it to Git (it's already in `.gitignore`)

## Verifying Your Setup

After creating the `.env` file, run:

```bash
python main.py
```

If you see "Error: OpenAI API key not configured!", check that:
1. The `.env` file is in the same directory as `main.py`
2. The file is named exactly `.env` (with the dot)
3. Your API key is correct and starts with `sk-`

## Cost Information

The application uses the `gpt-4o-mini` model by default, which is very cost-effective:
- Typical cost per quiz session: $0.01 - $0.05
- 25 questions per subtopic, 5 subtopics = ~$0.10 for initial generation

You can change the model in `config.py` if you need higher quality questions.

