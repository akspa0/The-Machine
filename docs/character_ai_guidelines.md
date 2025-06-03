# Character.AI Creation Guidelines

## Table of Contents
- [Introduction](#introduction)
- [Character Attributes Overview](#character-attributes-overview)
- [Quick Creation Guide](#quick-creation-guide)
- [Core Character Attributes](#core-character-attributes)
  - [Name](#name)
  - [Avatar](#avatar)
  - [Greeting](#greeting)
  - [Short Description](#short-description)
  - [Long Description](#long-description)
  - [Suggested Starters](#suggested-starters)
  - [Categories](#categories)
- [Visibility Settings](#visibility-settings)
  - [Character Visibility](#character-visibility)
  - [Remixing and Definition Visibility](#remixing-and-definition-visibility)
- [Image Generation Settings](#image-generation-settings)
  - [Image Generation](#image-generation)
  - [Image Style](#image-style)
  - [Direct Image Description Visibility](#direct-image-description-visibility)
- [Advanced Character Creation](#advanced-character-creation)
  - [Definition](#definition)
  - [Example Conversations](#example-conversations)
  - [Dialog Definitions](#dialog-definitions)
  - [Variables in Dialog](#variables-in-dialog)
- [Advanced Techniques](#advanced-techniques)
  - [Setting a Scene](#setting-a-scene)
  - [Negative Guidance](#negative-guidance)
  - [Using Formatting](#using-formatting)
  - [Multiple Characters in One Definition](#multiple-characters-in-one-definition)
- [Best Practices](#best-practices)
  - [Character Persona Development](#character-persona-development)
  - [Dialog Examples](#dialog-examples)
  - [JSON Formatting](#json-formatting)
- [Templates](#templates)
  - [Basic Character Template](#basic-character-template)
  - [Advanced Character Template](#advanced-character-template)

## Introduction

Character.AI allows you to create and interact with AI characters of any kind. These characters can be anyone or anything - from historical figures to fictional characters, original creations, or even inanimate objects given personality.

This guide provides comprehensive information on creating engaging and effective characters on the Character.AI platform. Whether you're making your first character or looking to enhance your character creation skills, this guide covers everything from basic requirements to advanced techniques.

## Character Attributes Overview

Character behavior is influenced by four key factors:

1. **Character Attributes** - The information you provide when creating the character
2. **Character Training** - Feedback from conversations (star ratings)
3. **User Personas** - User-set preferences and details
4. **Conversation Context** - The current interaction

This guide primarily focuses on Character Attributes that you can define when creating your character.

## Quick Creation Guide

To quickly create a character:

1. Click on **+ Create** in the navigation bar
2. Select **Create a Character**
3. At minimum, provide:
   - **Name** - What your character will be called
   - **Greeting** - The first message your character will say
   - **Avatar** (optional but recommended) - Visual representation

After creating, you can test your character by starting a conversation and refine it by editing additional attributes.

## Core Character Attributes

### Name

**Required**

The name is how users will refer to your character in chat and how they'll discover them in search. For well-known characters (like "Albert Einstein"), the name alone carries significant context. For generic names (like "Mary"), other attributes become more important in defining the character.

### Avatar

The image that appears next to your character in conversations. You can:
- Generate an AI image
- Upload your own image

While avatars don't currently influence character behavior, they help with user recognition and engagement.

### Greeting

**Required in Quick Create**

The first message your character sends when starting a new conversation. The greeting serves multiple purposes:
- Establishes character identity
- Sets the scene or scenario
- Introduces gameplay or interaction mechanics
- Engages the user to respond

You can include the user's name in the greeting with `{{user}}` variable.

Example: `Hello {{user}}, I'm delighted to meet you! Would you like to discuss physics or perhaps hear about my theory of relativity?`

If left blank in advanced creation, the user will be prompted to speak first.

### Short Description

A brief description (tagline) that appears under your character on the homepage. This should concisely capture the essence of your character in a few words.

Example: `Theoretical physicist, Nobel Prize winner, and developer of the theory of relativity`

### Long Description

A more detailed description of your character that users can view in the character profile. This can include:
- Background information
- Personality traits
- Special abilities or knowledge
- Historical context
- Relationship to other characters or the world

### Suggested Starters

Conversation prompts that users can click to begin interacting with your character. Good starters:
- Reflect your character's interests and expertise
- Provide clear paths for engagement
- Invite diverse interactions
- Set up interesting scenarios

Example starters for Einstein:
- `Can you explain the theory of relativity in simple terms?`
- `What was your most important discovery?`
- `If you were alive today, what would you think of modern physics?`

### Categories

Tags that help users find your character when browsing or searching. While categories don't affect character behavior, they improve discoverability.

## Visibility Settings

### Character Visibility

Controls who can find and interact with your character:
- **Public** - Visible on Home and in search results
- **Unlisted** - Only accessible via direct link
- **Private** - Only you can access

### Remixing and Definition Visibility

Controls whether others can see and build upon your character's definition:
- **Public Definition** - Others can view and remix your character's settings
- **Private Definition** - Others can chat but cannot see detailed settings

## Image Generation Settings

### Image Generation

Enables your character to generate images in response to certain prompts or contexts. Characters with image generation are marked with 🎨.

### Image Style

Defines the visual aesthetic when your character generates images. Options include:
- Realistic
- Anime
- Digital Art
- Various artistic styles

### Direct Image Description Visibility

Controls whether users can see the text prompts used to generate images:
- **Visible** - Users can see image generation prompts
- **Hidden** - Users only see the resulting images

## Advanced Character Creation

### Definition

A large free-form field (up to 32,000 characters) that can contain:
- Structured example dialogs
- Character background
- Personality traits
- Knowledge and capabilities
- Behavioral guidelines

The Definition is the most powerful tool for character development, allowing for detailed customization beyond the basic attributes.

### Example Conversations

Pre-written conversations that show how your character should interact. These serve as behavioral examples for the AI to follow.

To add example conversations:
1. From the Edit Character page, click "Insert a chat"
2. Have a conversation with your character
3. Select/edit the conversation
4. Save it to your character's Definition

### Dialog Definitions

The syntax for dialog in the Definition is:
```
Name: Message content
```

Example:
```
Einstein: E=mc² is perhaps my most famous equation. It demonstrates that energy and mass are interchangeable.
User: What does that mean in practical terms?
Einstein: Think of it this way - it shows that a tiny amount of matter can release an enormous amount of energy. That's the principle behind nuclear reactions!
```

### Variables in Dialog

Use these variables in your Definition to make it more flexible:
- `{{char}}` - Replaced with your character's name
- `{{user}}` - Replaced with the current user's name
- `{{random_user_1}}`, `{{random_user_2}}`, etc. - Replaced with consistent random names

Example with variables:
```
{{char}}: Welcome! I'm excited to discuss physics with you today.
{{random_user_1}}: Can you tell me about quantum mechanics?
{{char}}: Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles. It's quite fascinating, {{random_user_1}}!
```

## Advanced Techniques

### Setting a Scene

Establish a specific setting or scenario for your character by including descriptive scene-setting in the Definition.

Example:
```
[Setting: A 1920s Princeton University office filled with chalkboards covered in equations. {{char}} is seated at a cluttered desk, papers scattered around him, his wild white hair and mustache instantly recognizable.]

{{char}}: *looking up from his notes* Ah, welcome to Princeton! It's not often I get visitors these days. What brings you to my humble office?
```

### Negative Guidance

Include information about what your character should NOT do to prevent unwanted behaviors.

Example:
```
{{char}} should NOT:
- Use complex mathematical equations without explanation
- Be condescending when explaining scientific concepts
- Claim knowledge of scientific discoveries after 1955
```

### Using Formatting

You can use various formatting techniques in your Definition:
- **Emojis** to express emotion
- **Markdown** for text styling (bold, italic)
- **Parenthetical comments** to indicate thoughts or actions
- **Multiple languages** if your character is multilingual

Example:
```
{{char}}: *adjusts glasses* I find this quite **fascinating**! (This is exactly the kind of question I enjoy)
```

### Multiple Characters in One Definition

For interactive scenarios with multiple characters, you can define additional personalities:

```
{{char}}: I propose we look at this problem from a theoretical perspective.
Niels Bohr: But Albert, we must consider the experimental evidence first!
{{char}}: *smiles and shakes head* Niels, always the experimentalist. Perhaps we can meet in the middle?
```

## Best Practices

### Character Persona Development

1. **Be specific** - Clear, detailed traits create more consistent characters
2. **Provide examples** - Show don't just tell how your character behaves
3. **Create depth** - Include quirks, flaws, and unique characteristics
4. **Consider context** - Think about how your character responds in different situations
5. **Limit scope** - Better to excel in a specific domain than be vague about many

### Dialog Examples

1. Make dialog examples demonstrate the traits you've defined
2. Include a variety of scenarios and emotional states
3. Show how your character handles different types of questions
4. Demonstrate unique speech patterns or vocabulary
5. Include both verbal responses and physical actions/reactions

### JSON Formatting

For advanced users, structuring your Definition in JSON format can improve character consistency:

```json
{
  "{{char}}": {
    "name": "Albert Einstein",
    "personality": ["brilliant", "curious", "humble", "absent-minded", "pacifist"],
    "speech_style": {
      "tone": "Thoughtful and patient",
      "language": "Clear but occasionally uses analogies",
      "communication_style": "Explains complex ideas with simple examples"
    },
    "knowledge": ["physics", "relativity", "philosophy of science"],
    "background": "Born in Germany in 1879, developed the theory of relativity, won Nobel Prize in Physics, emigrated to the US in 1933"
  }
}
```

## Templates

### Basic Character Template

```
Name: [Character Name]

Short Description: [One-line description]

Greeting: Hello {{user}}! [Introduce character and invite interaction]

Long Description:
[Character Name] is [key traits]. Born in/from [origin], they [significant background details]. They are known for [notable achievements or abilities]. [Additional personality information].

Definition:
{{char}}'s personality: [list personality traits]
{{char}}'s appearance: [physical description]
{{char}}'s background: [brief history]

{{char}}: [Example greeting or typical statement]
{{random_user_1}}: [Example question]
{{char}}: [Example response showing personality]
```

### Advanced Character Template

For more detailed characters, you can use a JSON-structured template:

```json
{
  "{{char}}": {
    "name": "[Full Name]",
    "nickname": "[Nickname if applicable]",
    "age": "[Age]",
    "gender": "[Gender]",
    "physical_attributes": {
      "appearance": "[General appearance]",
      "height": "[Height]",
      "build": "[Body type]",
      "hair": "[Hair description]",
      "eyes": "[Eye description]",
      "distinguishing_features": "[Any unique physical traits]"
    },
    "attire": "[Typical clothing]",
    "personality_attributes": {
      "personality": ["[Trait 1]", "[Trait 2]", "[Trait 3]"],
      "likes": ["[Like 1]", "[Like 2]"],
      "dislikes": ["[Dislike 1]", "[Dislike 2]"],
      "strengths": ["[Strength 1]", "[Strength 2]"],
      "weaknesses": ["[Weakness 1]", "[Weakness 2]"]
    },
    "background": "[Character history]",
    "speech_style": {
      "tone": "[How they sound]",
      "mannerisms": "[Verbal habits]",
      "vocabulary": "[Word choice patterns]"
    },
    "instructions": "Interact with {{user}} as [Character Name], responding with [specific guidance]."
  }
}
```

---

This guide covers the essential aspects of character creation on Character.AI. Remember that character development is an iterative process - test your character in conversations and refine based on how it performs. The best characters evolve over time through feedback and refinement.

For the latest updates and features, check the official [Character.AI documentation](https://book.character.ai/). 