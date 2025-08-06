import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert'; // For JSON encoding/decoding
import 'package:url_launcher/url_launcher.dart'; // For opening URLs
import 'dart:io'; // Add this import for Platform checks

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Medi Guide',
      theme: ThemeData(
        primarySwatch: Colors.blueGrey,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        fontFamily: 'Inter',
        useMaterial3: true,
      ),
      home: const DiseasePredictorScreen(),
    );
  }
}

class DiseasePredictorScreen extends StatefulWidget {
  const DiseasePredictorScreen({super.key});

  @override
  State<DiseasePredictorScreen> createState() => _DiseasePredictorScreenState();
}

class _Message {
  final String text;
  final bool isUser;
  _Message(this.text, this.isUser);
}

class _DiseasePredictorScreenState extends State<DiseasePredictorScreen> {
  final TextEditingController _symptomsController = TextEditingController();
  final TextEditingController _genderController = TextEditingController();
  String _predictedDisease = 'Enter symptoms to get a prediction!';
  List<Map<String, dynamic>> _drugInformation = [];
  bool _isLoading = false;
  final List<_Message> _messages = [
    _Message("Hi! I'm Medi Guide. Describe your symptoms and I'll try to help.",
        false),
  ];
  final ScrollController _scrollController = ScrollController();

  // The URL of your FastAPI endpoint for prediction
  late final String _predictApiEndpoint;
  // The URL of your FastAPI endpoint for drug information
  final String _drugInfoApiEndpoint = 'http://10.0.2.2:8000/get_drug_info';

  @override
  void initState() {
    super.initState();
    // Set API endpoint based on platform
    if (Platform.isAndroid) {
      _predictApiEndpoint = 'http://10.0.2.2:8000/predict';
    } else {
      _predictApiEndpoint = 'http://127.0.0.1:8000/predict';
    }
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _symptomsController.dispose();
    _genderController.dispose();
    super.dispose();
  }

  Future<void> _predictDisease() async {
    setState(() {
      _isLoading = true;
      _predictedDisease = 'Predicting...';
      _drugInformation = []; // Clear previous drug info
    });

    final String symptoms = _symptomsController.text.trim();
    final String gender = _genderController.text.trim();

    if (symptoms.isEmpty) {
      setState(() {
        _predictedDisease = 'Please enter symptoms.';
        _isLoading = false;
      });
      return;
    }

    try {
      final response = await http.post(
        Uri.parse(_predictApiEndpoint), // Use prediction endpoint
        headers: <String, String>{
          'Content-Type': 'application/json',
        },
        body: jsonEncode(<String, dynamic>{
          'symptoms': symptoms,
          'gender': 'male',
        }),
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = jsonDecode(response.body);
        String disease = responseData['predicted_disease'];
        setState(() {
          _predictedDisease = 'Predicted Disease: $disease';
        });
        // After successful prediction, fetch drug information
        //await _getDrugInformation(disease);
      } else {
        final Map<String, dynamic> errorData = jsonDecode(response.body);
        setState(() {
          _predictedDisease =
              'Error: ${response.statusCode} - ${errorData['detail'] ?? 'Unknown error'}';
        });
        print('API Error: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      setState(() {
        _predictedDisease =
            'Failed to connect to Prediction API. Is the server running? Error: $e';
      });
      print('Prediction API Call Error: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _getDrugInformation(String diseaseName) async {
    try {
      final response = await http.post(
        Uri.parse(_drugInfoApiEndpoint), // Use drug info endpoint
        headers: <String, String>{
          'Content-Type': 'application/json',
        },
        body: jsonEncode(<String, dynamic>{
          'disease_name': diseaseName,
        }),
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = jsonDecode(response.body);
        setState(() {
          // Ensure 'drug_information' key exists and is a list
          if (responseData.containsKey('drug_information') &&
              responseData['drug_information'] is List) {
            _drugInformation = List<Map<String, dynamic>>.from(
                responseData['drug_information']);
          } else {
            _drugInformation = [
              {
                'snippet': 'No specific drug information found.',
                'source_title': '',
                'url': ''
              }
            ];
          }
        });
      } else {
        final Map<String, dynamic> errorData = jsonDecode(response.body);
        setState(() {
          _drugInformation = [
            {
              'snippet':
                  'Failed to load drug info: ${response.statusCode} - ${errorData['detail'] ?? 'Unknown error'}',
              'source_title': '',
              'url': ''
            }
          ];
        });
        print('Drug Info API Error: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      setState(() {
        _drugInformation = [
          {
            'snippet': 'Failed to connect to Drug Info API. Error: $e',
            'source_title': '',
            'url': ''
          }
        ];
      });
      print('Drug Info API Call Error: $e');
    }
  }

  // Function to launch URL
  Future<void> _launchUrl(String url) async {
    if (url.isEmpty) return;
    final Uri uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri);
    } else {
      // Handle the case where the URL cannot be launched
      print('Could not launch $uri');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not open link: $url')),
      );
    }
  }

  Future<void> _sendMessage() async {
    final text = _symptomsController.text.trim();
    if (text.isEmpty) return;
    setState(() {
      _messages.add(_Message(text, true));
      _isLoading = true;
      _symptomsController.clear();
    });
    _scrollToBottom();

    // Simulate API call
    try {
      final response = await http.post(
        Uri.parse(_predictApiEndpoint),
        headers: <String, String>{'Content-Type': 'application/json'},
        body: jsonEncode(<String, dynamic>{
          'symptoms': text,
        }),
      );
      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = jsonDecode(response.body);
        String disease =
            responseData['predictions']['Final Ensemble Prediction'];
        String medication = responseData['recommended_medication'];
        setState(() {
          _messages.add(_Message("Predicted Disease is:\n$disease", false));
          _messages
              .add(_Message("Predicted Medication's are:\n$medication", false));
        });
        // Optionally fetch drug info and append to chat
        // await _getDrugInformation(disease);
      } else {
        setState(() {
          _messages
              .add(_Message("Sorry, I couldn't process your request.", false));
        });
      }
    } catch (e) {
      setState(() {
        _messages.add(_Message("Failed to connect to the server.", false));
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Widget _buildMessageBubble(_Message message) {
    final isUser = message.isUser;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: EdgeInsets.symmetric(vertical: 4, horizontal: 8),
        padding: EdgeInsets.all(14),
        constraints:
            BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.75),
        decoration: BoxDecoration(
          color: isUser
              ? Theme.of(context).colorScheme.onPrimaryContainer
              : Colors.blueGrey.shade100,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(18),
            topRight: Radius.circular(18),
            bottomLeft: Radius.circular(isUser ? 18 : 0),
            bottomRight: Radius.circular(isUser ? 0 : 18),
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black12,
              blurRadius: 2,
              offset: Offset(0, 1),
            ),
          ],
        ),
        child: Text(
          message.text,
          style: TextStyle(
            color: isUser ? Colors.white : Colors.black87,
            fontSize: 16,
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Medi Guide'),
        centerTitle: true,
        backgroundColor: Theme.of(context).colorScheme.onPrimaryContainer,
        foregroundColor: Colors.white,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(bottom: Radius.circular(20)),
        ),
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: ListView.builder(
                controller: _scrollController,
                reverse: false,
                padding: EdgeInsets.symmetric(vertical: 16),
                itemCount: _messages.length,
                itemBuilder: (context, index) {
                  final message = _messages[index];
                  return Row(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    mainAxisAlignment: message.isUser
                        ? MainAxisAlignment.end
                        : MainAxisAlignment.start,
                    children: [
                      if (!message.isUser)
                        Padding(
                          padding:
                              const EdgeInsets.only(left: 12.0, right: 4.0),
                          child: CircleAvatar(
                            backgroundColor: Colors.blueGrey.shade500,
                            child: Icon(Icons.medical_services,
                                color: Colors.white),
                          ),
                        ),
                      Flexible(child: _buildMessageBubble(message)),
                      if (message.isUser)
                        Padding(
                          padding:
                              const EdgeInsets.only(right: 12.0, left: 4.0),
                          child: CircleAvatar(
                            backgroundColor: Theme.of(context)
                                .colorScheme
                                .onPrimaryContainer,
                            child:
                                Icon(Icons.person_search, color: Colors.white),
                          ),
                        ),
                    ],
                  );
                },
              ),
            ),
            if (_isLoading)
              Padding(
                padding: const EdgeInsets.only(bottom: 8.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.start,
                  children: [
                    Padding(
                      padding: const EdgeInsets.only(left: 20.0, right: 8.0),
                      child: CircleAvatar(
                        backgroundColor: Colors.blueGrey.shade200,
                        child:
                            Icon(Icons.medical_services, color: Colors.white),
                      ),
                    ),
                    Container(
                      padding: EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.blueGrey.shade50,
                        borderRadius: BorderRadius.circular(18),
                      ),
                      child: SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                    ),
                  ],
                ),
              ),
            Container(
              color: Colors.white,
              padding: EdgeInsets.fromLTRB(12, 8, 12, 12),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _symptomsController,
                      decoration: InputDecoration(
                        hintText: 'Describe your symptoms...',
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(24),
                          borderSide: BorderSide.none,
                        ),
                        filled: true,
                        fillColor: Colors.blueGrey.shade50,
                        contentPadding:
                            EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                      ),
                      onSubmitted: (_) => _sendMessage(),
                    ),
                  ),
                  SizedBox(width: 8),
                  FloatingActionButton(
                    onPressed: _isLoading ? null : _sendMessage,
                    backgroundColor:
                        Theme.of(context).colorScheme.onPrimaryContainer,
                    elevation: 2,
                    mini: true,
                    child: Icon(Icons.send, color: Colors.white),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
