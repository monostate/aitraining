class Seq2SeqDataset:
    """
    A dataset class for sequence-to-sequence tasks.

    Args:
        data (list): The dataset containing input and target sequences.
        tokenizer (PreTrainedTokenizer): The tokenizer to process the text data.
        config (object): Configuration object containing dataset parameters.

    Attributes:
        data (list): The dataset containing input and target sequences.
        tokenizer (PreTrainedTokenizer): The tokenizer to process the text data.
        config (object): Configuration object containing dataset parameters.
        max_len_input (int): Maximum length for input sequences.
        max_len_target (int): Maximum length for target sequences.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(item): Returns the tokenized input and target sequences for a given index.
    """

    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.max_len_input = self.config.max_seq_length
        self.max_len_target = self.config.max_target_length

        # Validate required columns exist
        if len(self.data) > 0:
            # Check if data has column_names attribute (datasets library)
            if hasattr(self.data, "column_names"):
                columns = self.data.column_names
            elif hasattr(self.data, "columns"):
                columns = self.data.columns
            else:
                # For list-like data, check the first item
                columns = list(self.data[0].keys()) if isinstance(self.data[0], dict) else []

            # Validate text column exists
            if columns and self.config.text_column not in columns:
                raise KeyError(
                    f"Required column '{self.config.text_column}' not found in dataset. Available columns: {columns}"
                )

            # Validate target column exists
            if columns and self.config.target_column not in columns:
                raise KeyError(
                    f"Required column '{self.config.target_column}' not found in dataset. Available columns: {columns}"
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = str(self.data[item][self.config.text_column])
        target = str(self.data[item][self.config.target_column])

        model_inputs = self.tokenizer(text, max_length=self.max_len_input, truncation=True)

        labels = self.tokenizer(text_target=target, max_length=self.max_len_target, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
