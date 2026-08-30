from llm.openai_llm import OpenAILLM


def main():

    print("=" * 70)

    print(
        "SBI INSURANCE POLICY - "
        "OPENAI LLM TEST"
    )

    print("=" * 70)

    prompt = """
You are an AI assistant specialized in
SBI insurance policy documents.

Answer the following question using only
the information provided below.

Policy Context:

Accident and Sickness Medical expenses are
covered for medically necessary medical
expenses incurred overseas for medical
treatment due to disease, illness, or injury
first sustained or contracted during the
insured trip.

The expenses covered include physician
services, hospital services, medically
necessary services, and local emergency
medical transportation.

Question:

What medical expenses are covered?

Answer clearly and concisely.
"""

    llm = OpenAILLM()

    answer = llm.generate(prompt)

    print("\n" + "=" * 70)

    print("LLM RESPONSE")

    print("=" * 70)

    print(answer)

    print("\n" + "=" * 70)


if __name__ == "__main__":

    main()