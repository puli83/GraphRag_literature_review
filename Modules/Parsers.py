import re

class ParserLLM_KG_extracted_output:
    """
    A class to parse and clean the output of an LLM, specifically
    to extract 'think' sections and structured 'relationships'.
    """

    def __init__(self):
        """
        Initializes the ParserLLM_KG_extracted_output.
        No specific initialization parameters are needed for this version.
        """
        # Define the complex regex for relationship keywords here for easier management
        self.relationship_keywords_pattern = (
            r'(RELATIONSHIPS|RELATIONSHMENTS|RELATIONSHIONS|'
            r'RELATIONSHONES|RELATIONSHINES|RELATIONSHATIONS)'
        )
        # Define the end tags pattern
        self.end_tags_pattern = r'(<<</SYS>>|\[INST\]|<<SYS>>|\[/INST\])'


    def extract_chain_of_thoughts(self, input_string: str) -> str:
        """
        Extracts the content between <think> and </think> tags.
        """
        start_tag = "<think>"
        end_tag = "</think>"

        start_index = input_string.find(start_tag)
        if start_index == -1:
            return "" # Start tag not found

        start_index += len(start_tag)
        end_index = input_string.find(end_tag, start_index)

        if end_index == -1:
            return "" # End tag not found after start tag

        think_part = input_string[start_index:end_index].strip()
        return think_part

    def _add_newline_before_tag(self, text: str) -> str:
        """
        Adds a newline before specific system tags if they are not at the beginning of a line.
        Internal helper method.
        """
        # This regex looks for any character (non-greedy) followed by one of the tags,
        # and inserts a newline before the tag. It only does this for the first occurrence.
        return re.sub(r'(.)(' + self.end_tags_pattern + r')', r'\1\n\2', text, count=1)


    def clean_response_llm(self, text: str) -> tuple[str, str]:
        """
        Cleans the LLM response to isolate the 'think' part and the main 'relationships' content.
        """
        if not isinstance(text, str):
            print("Warning: Input text is not a string.")
            return "", ""

        think_part = self.extract_chain_of_thoughts(text)

        pattern_with_end_tag = re.compile(
            self.relationship_keywords_pattern + r'.*?' + self.end_tags_pattern,
            re.DOTALL | re.IGNORECASE
        )

        pattern_without_end_tag = re.compile(
            self.relationship_keywords_pattern + r'.*',
            re.DOTALL | re.IGNORECASE
        )

        result_match = pattern_with_end_tag.search(text)

        extracted_content = ""
        if result_match:
            extracted_content = result_match.group(0)
            extracted_content = re.split(self.end_tags_pattern, extracted_content, maxsplit=1, flags=re.IGNORECASE)[0]
        else:
            result_match_no_tag = pattern_without_end_tag.search(text)
            if result_match_no_tag:
                extracted_content = result_match_no_tag.group(0)

        if not extracted_content:
            return think_part, ""

        return think_part, extracted_content.strip()


    def parse_text(self, text: str) -> tuple[str, dict]:
        """
        Parses the cleaned LLM text to extract thinking steps and structured relationships.

        Args:
            text (str): The raw output string from the LLM.

        Returns:
            tuple[str, dict]: A tuple containing the thinking text and a dictionary
                              with a "Relationships" key (list of strings).
        """
        if not isinstance(text, str):
            print("Error: Input to parse_text must be a string.")
            return "", {"Relationships": []}

        thinking_text, cleaned_text_for_relationships = self.clean_response_llm(text)

        result_dict = {"Relationships": []}

        if not cleaned_text_for_relationships:
            return thinking_text, result_dict

        try:
            relationships_section_parts = re.split(self.relationship_keywords_pattern + r":", cleaned_text_for_relationships, maxsplit=1, flags=re.IGNORECASE)

            if len(relationships_section_parts) > 1:
                relationships_data = relationships_section_parts[-1]
                relationships = [r.strip() for r in relationships_data.strip().split('\n') if r.strip()]

                valid_relationships = set()
                for rel in relationships:
                    if re.search('->|→', rel): # Ensure it looks like a relationship
                        valid_relationships.add(rel)
                result_dict["Relationships"] = sorted(list(valid_relationships))
            else:
                pass # No colon after keyword or empty section

        except Exception as e:
            print(f"Error processing relationships section: {e} in text '{cleaned_text_for_relationships[:100]}...'")
            result_dict["Relationships"] = []

        return thinking_text, result_dict

    def extract_structured_relationships(self, parsed_data: dict) -> list[dict]:
        """
        Extracts structured entities (source, target, relation) from the list of
        relationship strings obtained from parse_text.

        Args:
            parsed_data (dict): The dictionary output from parse_text, specifically
                                containing the "Relationships" key with a list of strings.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary represents a
                        structured relationship with 'source', 'target', and 'relation' keys.
        """
        structured_relationships = []
        if "Relationships" not in parsed_data or not isinstance(parsed_data["Relationships"], list):
            return structured_relationships

        for line in parsed_data['Relationships']:
            # Initialize default values
            # weight = None # Weight extraction is commented out as per user's original snippet
            relation = ""
            source = ""
            target = ""

            # Extract weight using regex (commented out)
            # weight_match = re.search(r'\[.*?([0-3])\]', line)
            # weight_match = re.search(r'\[(\d+)\]', line)
            # if weight_match:
            #     weight = int(weight_match.group(1))
            #     main_part = line.split('[')[0]
            # else:
            #     main_part = line

            main_part = line # Use the full line as main_part since weight is not extracted

            # Extract relation type from parentheses
            relation_match = re.search(r'\((.*?)\)', main_part) # Search in main_part
            if relation_match:
                relation = relation_match.group(1).strip()
                # Remove the relation part from main_part to correctly isolate source and target
                main_part = re.split(r'\(.*?\)', main_part, maxsplit=1)[0].strip()

            # Split on arrow (handling both -> and →)
            parts = re.split(r'\s*->\s*|\s*→\s*', main_part) # Add \s* for robustness around arrows
            if len(parts) >= 2:
                source = parts[0].strip()
                # Target is the last part if there are multiple arrows (though typically not expected)
                target = parts[-1].strip()

                # If relation was not in parentheses, it might be the middle part for A->B->C cases
                # For A -> B (relation) format, this is already handled.
                # For A -> relation -> B format (less common with current parser), this would need adjustment.
                # The current logic prioritizes relation in parentheses.

                if source and target: # Ensure both source and target were found
                    structured_relationships.append({
                        "source": source,
                        "target": target,
                        "relation": relation if relation else "unknown" # Default if no relation in ()
                        # "weight": weight # If weight were to be included
                    })
            elif len(parts) == 1 and parts[0].strip() != "": # Case where there's text but no arrow
                # This might be a single entity or malformed relationship.
                # Decide how to handle: skip, or log, or add as a single node if relevant.
                # For now, we skip if it doesn't split into at least two parts.
                # print(f"Skipping malformed relationship line: {line}")
                pass


        return structured_relationships

# --- Example Usage ---
if __name__ == "__main__":
    parser = ParserLLM_KG_extracted_output()

    sample_text_1 = """
    [INST] Extract relationships:
    <think>
    The user wants to extract relationships.
    I need to identify entities and how they are connected.
    - Entity A is related to Entity B.
    - Concept X impacts Concept Y.
    </think>
    RELATIONSHIPS:
    Entity A -> Entity B (is related to)
    Concept X -> Concept Y (impacts)
    Entity C -> Entity D (testing)
    <<</SYS>>
    """

    sample_text_2 = """
    <think>Okay, I will find the connections.</think>
    RELATIONSHIONS:
    Wireless Communication -> Speed (improved in)
    5G -> Previous Generations (compared with)
    [INST] Any other details?
    """

    sample_text_3 = """
    <think>This one has no clear end tag for relationships but should still work.</think>
    RELATIONSHMENTS:
    Alpha -> Beta (connects)
    Gamma -> Delta (supports)
    Alpha -> Beta (connects)
    """

    sample_text_4 = "No think tags, no relationship tags."

    sample_text_5 = "<think>Only thinking</think>"

    sample_text_6 = """
    RELATIONSHIPS:
    Valid -> Arrow (has)
    No Arrow Here
    Another -> Valid (is a)
    """

    sample_text_7 = """
    <think>
    The user is asking for relationships from the provided text.
    I will identify entities and their connections.
    For example, "5G technology" is an entity, "smartphones" is another, and "used in" is the relationship.
    </think>
    Okay, here are the relationships:
    RELATIONSHIPS:
    5G technology -> smartphones (used in)
    Mobile Companies in China -> 5G Smartphones (launch)
    Wireless Communication -> Speed (evolve with)
    """

    sample_text_8 = """
    RELATIONSHIPS:
    Source Only -> (has relation)
    -> Target Only (is part of)
    Source Two -> Target Two
    """


    tests = [sample_text_1, sample_text_2, sample_text_3, sample_text_4, sample_text_5, sample_text_6, sample_text_7, sample_text_8]

    for i, text_input in enumerate(tests):
        print(f"\n--- Test Case {i+1} ---")
        thinking, relations_dict = parser.parse_text(text_input)
        print(f"Thinking:\n'{thinking}'")
        print(f"Parsed Relationships Dict:\n{relations_dict}")

        # Now test the new method
        structured_rels = parser.extract_structured_relationships(relations_dict)
        print(f"Structured Relationships Extracted ({len(structured_rels)}):")
        if structured_rels:
            for rel_item in structured_rels:
                print(f"  - {rel_item}")
        else:
            print("  - None")


    print("\n--- Test with None input ---")
    thinking_none, relations_none = parser.parse_text(None)
    print(f"Thinking (None input):\n'{thinking_none}'")
    print(f"Relationships Dict (None input):\n{relations_none}")
    structured_none = parser.extract_structured_relationships(relations_none)
    print(f"Structured Relationships (None input):\n{structured_none}")

