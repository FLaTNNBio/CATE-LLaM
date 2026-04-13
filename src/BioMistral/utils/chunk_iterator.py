class ChunkIterator:

    def __init__(self, df, chunk_size, processed_indices):
        self.df = df
        self.chunk_size = chunk_size
        self.processed_indices = processed_indices

    def __iter__(self):

        for start in range(0, len(self.df), self.chunk_size):

            chunk = self.df.iloc[start:start + self.chunk_size]
            chunk = chunk[~chunk.index.isin(self.processed_indices)]

            if not chunk.empty:
                yield chunk