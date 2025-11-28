# JavaScript Unit Tests for AutoTrain Web UI

This directory contains JavaScript unit tests for the AutoTrain web UI form renderer.

## Setup

Install dependencies:

```bash
cd tests/app/js
npm install
```

## Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode (auto-rerun on file changes)
npm run test:watch

# Run tests with coverage report
npm run test:coverage
```

## Test Files

- `ui-helpers.js` - Extracted helper functions from listeners.js for testing
- `ui-helpers.test.js` - Unit tests for the helper functions

## What's Tested

### Parameter Grouping
- `groupParameters()` - Groups parameters by their group metadata
- Handles parameters without groups
- Creates separate grouped and ungrouped collections

### Advanced Panel Logic
- `getAdvancedGroups()` - Identifies which groups belong in advanced panel
- `getGroupOrder()` - Returns consistent group ordering
- Verifies basic groups appear before advanced groups

### PPO Validation Logic
- `isPPOTask()` - Detects when PPO trainer is selected
- `shouldEnablePPOControls()` - Determines if PPO controls should be enabled
- `getPPOControlState()` - Calculates state for individual PPO controls
- Validates requirement field logic

### HTML Generation
- `createParameterHTML()` - Generates form elements with proper attributes
- Includes help icons when help text is provided
- Marks requirement fields with asterisks
- Handles different input types (number, string, dropdown, checkbox, textarea)

## Coverage Goals

Target: >90% code coverage for all helper functions

Current test coverage includes:
- ✅ Parameter grouping logic
- ✅ Advanced panel categorization
- ✅ PPO validation workflow
- ✅ HTML element generation
- ✅ Edge cases (empty inputs, null values, etc.)

## Integration with Main Code

The `ui-helpers.js` file contains extracted, testable versions of the core logic from
`src/autotrain/app/static/scripts/listeners.js`. These functions can be imported and
used in the main code to ensure consistency between the implementation and tests.

## Future Enhancements

Potential additions:
- Mock DOM tests for full renderUI() function
- Tests for event listener attachment
- Tests for JSON synchronization
- Accessibility tests (ARIA labels, keyboard navigation)
